import torch
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter


class ModelTrainer:
    def __init__(self, image_active_sampler):
        self.opt_params = None
        self._image_active_sampler = image_active_sampler
        self.localization_poses = []

    def train_model(self,
                    model,
                    dataset_loader,
                    num_epochs,
                    is_image_active_sampling,
                    optimizer_params=None,
                    verbose=True):
        """
        :param model:
        :param dataset_loader:
        :param num_epochs:
        :param is_image_active_sampling:
        :param optimizer_params:  Default lr=0.005
        :param verbose:
        :return:
        """
        optimizer_params = ModelTrainer.check_optimizer_params(optimizer_params)
        if verbose:
            writer = SummaryWriter()
        model.requires_grad_(True)
        model.cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        for i in trange(num_epochs):
            for state in dataset_loader:
                loss = self.train(model, optimizer, state, is_image_active_sampling)
            ModelTrainer.log_losses(writer, loss, i, verbose=verbose)
            # trainer.reset_params()
            # clear_output(wait=True)
        del loss, optimizer
        torch.cuda.empty_cache()

    def localization(self,
                     model,
                     tracking_dataset_loader,
                     num_epochs=100,
                     is_image_active_sampling=False,
                     optimizer_params=None,
                     verbose=True):
        if verbose:
            writer = SummaryWriter()

        optimizer_params = ModelTrainer.check_optimizer_params(optimizer_params)

        self.localization_poses = []
        model.cuda()
        model.eval()
        model.requires_grad_(False)
        is_initialization = True
        for state in tqdm(tracking_dataset_loader):
            if is_initialization:
                is_initialization = False
            else:
                state.set_position(current_position)

            state.train_position()
            state._position.cuda()
            optimizer = torch.optim.Adam([state._position], **optimizer_params)
            self.reset_params()
            for i in range(num_epochs):
                loss = self.train(model, optimizer, state, is_image_active_sampling)
                ModelTrainer.log_losses(writer, loss, i, verbose=verbose)

            state.freeze_position()
            state._position.cpu()

            current_position = state.get_matrix_position().detach().numpy()
            self.localization_poses.append(current_position.copy())

        del state, loss, optimizer
        torch.cuda.empty_cache()
        return self.localization_poses

    def train(self, model, optimizer, state, is_image_active_sampling):
        self.load_optimizer_state(optimizer)
        optimizer.zero_grad()

        losses, data_batch = self.sample_and_backward_batch(state, state.frame.get_pixel_probs(), model)
        if is_image_active_sampling:
            new_pixel_weights = self._image_active_sampler.estimate_pixels_weights(
                data_batch['pixel'],
                losses.loss,
                state.frame.get_pixel_probs())

            losses, data_batch = self.sample_and_backward_batch(state, new_pixel_weights, model)

        optimizer.step()
        optimizer.zero_grad()
        self.save_optimizer_state(optimizer)

        return losses

    def sample_and_backward_batch(self, state, pixel_weights, model):
        data_batch = self.sample_batch(state, pixel_weights)
        return self.backward_batch(model, data_batch), data_batch

    def sample_batch(self, state, weights):
        y, x = self._image_active_sampler.sample_pixels(weights)
        data_batch = self._image_active_sampler.get_training_data(state, y, x)
        ModelTrainer.send_batch_to_model_device(data_batch)
        return data_batch

    def backward_batch(self, model, data_batch):
        losses = self.forward_model(model, data_batch)
        self.backward_mean_loss(model, losses)
        return losses

    def save_optimizer_state(self, optimizer):
        self.opt_params = optimizer.state_dict()

    def load_optimizer_state(self, optimizer):
        if self.opt_params is not None:
            optimizer.load_state_dict(self.opt_params)

    @staticmethod
    def forward_model(model, data_batch):
        output = model.forward(data_batch["pixel"], data_batch['camera_position'])
        return model.losses(output, data_batch['color'], data_batch['depth'])

    def reset_params(self):
        self.opt_params = None

    @staticmethod
    def backward_mean_loss(model, losses):
        mean_loss = model.mean_loss(losses.loss)
        mean_loss.backward()
        return mean_loss.item()

    @staticmethod
    def send_batch_to_model_device(batch, device='cuda'):
        batch['pixel'] = batch['pixel'].to(device)
        batch['color'] = batch['color'].to(device)
        batch['depth'] = batch['depth'].to(device)
        batch['camera_position'] = batch['camera_position'].to(device)

    @staticmethod
    def log_losses(writer, loss, i, verbose):
        if verbose:
            writer.add_scalar('image/coarse_image_loss', torch.mean(loss.coarse_image_loss).item(), i, new_style=True)
            writer.add_scalar('image/fine_image_loss', torch.mean(loss.fine_image_loss).item(), i, new_style=True)
            writer.add_scalar('depth/coarse_depth_loss', torch.mean(loss.coarse_depth_loss).item(), i, new_style=True)
            writer.add_scalar('depth/fine_depth_loss', torch.mean(loss.fine_depth_loss).item(), i, new_style=True)
            writer.add_scalar('loss', torch.mean(loss.loss).item(), i, new_style=True)

    @staticmethod
    def check_optimizer_params(optimizer_params):
        if optimizer_params is None:
            optimizer_params = {'lr': 0.005}
        return optimizer_params