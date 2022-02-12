import torch
from tqdm.auto import tqdm, trange
from imap.trainers.train_logger import TrainLogger

class ModelTrainer:
    def __init__(self, image_active_sampler, device='cuda'):
        self.opt_params = None
        self._image_active_sampler = image_active_sampler
        self.localization_poses = []
        self._device = device

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
        with TrainLogger('model_training') as logger:
            model.requires_grad_(True)
            model.cuda()
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
            for i in trange(num_epochs):
                for state in dataset_loader:
                    # self.load_optimizer_state(optimizer)
                    optimizer.zero_grad()
                    loss, mean_loss = self.backward_model(model, state, is_image_active_sampling)
                    optimizer.step()
                    # optimizer.zero_grad()
                    # self.save_optimizer_state(optimizer)
                logger.log_losses(mean_loss, i, verbose=verbose)

            optimizer.zero_grad()
            del loss, optimizer
        torch.cuda.empty_cache()

    # def localization(self,
    #                  model,
    #                  tracking_dataset_loader,
    #                  num_epochs=100,
    #                  is_image_active_sampling=False,
    #                  optimizer_params=None,
    #                  verbose=True):
    #     if verbose:
    #         writer = SummaryWriter()
    #
    #     optimizer_params = ModelTrainer.check_optimizer_params(optimizer_params)
    #
    #     self.localization_poses = []
    #     model.cuda()
    #     model.eval()
    #     model.requires_grad_(False)
    #     is_initialization = True
    #     for state in tqdm(tracking_dataset_loader):
    #         if is_initialization:
    #             is_initialization = False
    #         else:
    #             state.set_position(current_position)
    #
    #         state.train_position()
    #         state._position.cuda()
    #         optimizer = torch.optim.Adam([state._position], **optimizer_params)
    #         self.reset_params()
    #         for i in range(num_epochs):
    #             loss = self.train(model, optimizer, state, is_image_active_sampling)
    #             ModelTrainer.log_losses(writer, loss, i, verbose=verbose)
    #
    #         state.freeze_position()
    #         state._position.cpu()
    #
    #         current_position = state.get_matrix_position().detach().numpy()
    #         self.localization_poses.append(current_position.copy())
    #
    #     del state, loss, optimizer
    #     torch.cuda.empty_cache()
    #     return self.localization_poses

    def backward_model(self, model, state, is_image_active_sampling):
        # 2 backwards + image_active_sampling without grad
        # TODO: name losses like random and active & return both
        losses, data_batch, mean_losses = self.backward_batch(state, state.frame.get_pixel_probs(), model)
        if is_image_active_sampling:
            new_pixel_weights = self._image_active_sampler.estimate_pixels_weights(
                data_batch['pixel'],
                losses.loss,
                state.frame.get_pixel_probs())

            losses, data_batch, mean_losses = self.backward_batch(state, new_pixel_weights, model)

        return losses, mean_losses

    def backward_batch(self, state, pixel_weights, model):
        data_batch = self._image_active_sampler.sample_batch(state, pixel_weights, device=self._device)
        output = model.forward(data_batch["pixel"], data_batch['camera_position'])
        losses = model.losses(output, data_batch['color'], data_batch['depth'])
        mean_loss = losses.mean_loss()
        mean_loss.loss.backward()
        return losses, data_batch, mean_loss

    def save_optimizer_state(self, optimizer):
        self.opt_params = optimizer.state_dict()

    def load_optimizer_state(self, optimizer):
        if self.opt_params is not None:
            optimizer.load_state_dict(self.opt_params)

    def reset_params(self):
        self.opt_params = None

    @staticmethod
    def send_batch_to_model_device(batch, device='cuda'):
        batch['pixel'] = batch['pixel'].to(device)
        batch['color'] = batch['color'].to(device)
        batch['depth'] = batch['depth'].to(device)
        batch['camera_position'] = batch['camera_position'].to(device)

    @staticmethod
    def check_optimizer_params(optimizer_params):
        if optimizer_params is None:
            optimizer_params = {'lr': 0.005}
        return optimizer_params
