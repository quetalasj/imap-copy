import torch
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer:
    def __init__(self, parameters,  image_active_sampler, lr=0.005, **kwargs):
        self.optimizer = torch.optim.Adam(parameters, lr=lr, **kwargs)
        self.opt_params = None
        self._image_active_sampler = image_active_sampler

    def train_model(self,
                    model,
                    dataset_loader,
                    camera,
                    num_epochs,
                    is_image_active_sampling,
                    verbose=True):
        if verbose:
            writer = SummaryWriter()
        model.requires_grad_(True)
        for i in trange(num_epochs):
            for color_image, depth_image, position in dataset_loader:
                state = camera.create_state(color_image, depth_image, position)
                loss = self.train(model, state, is_image_active_sampling)
            ModelTrainer.log_losses(writer, loss, i, verbose=verbose)
            # trainer.reset_params()
            # clear_output(wait=True)
        del loss
        torch.cuda.empty_cache()

    def localization(self,
                     model,
                     tracking_dataset_loader,
                     camera,
                     num_epochs=100,
                     is_image_active_sampling=False,
                     verbose=True):
        if verbose:
            writer = SummaryWriter()

        poses = []
        model.cuda()
        model.eval()
        model.requires_grad_(False)
        is_initialization = True

        for color_image, depth_image, p in tqdm(tracking_dataset_loader):
            if is_initialization:
                current_position = p
                state = camera.create_state(color_image, depth_image, current_position, process_position=True)
                state.train_position()
                state._position.cuda()
                self.optimizer = torch.optim.Adam([state._position], lr=0.005)
                is_initialization = False
            else:
                state = camera.create_state(color_image, depth_image, current_position, process_position=False)
                state.train_position()
                state._position.cuda()
                self.optimizer.add_param_group({'params': state._position})

            self.reset_params()

            for i in trange(num_epochs):
                loss = self.train(model, state, is_image_active_sampling)
                ModelTrainer.log_losses(writer, loss, i, verbose=verbose)

            state.freeze_position()
            state._position.cpu()

            current_position = state.get_matrix_position().detach().numpy()
            poses.append(current_position.copy())

        torch.cuda.empty_cache()
        return poses

    def train(self, model, state, is_image_active_sampling):
        """
        Train the model one epoch on batch of data
        :param model:
        :param data_batch: {
                            "pixel": np.array([x, y], dtype=np.float32),
                            "color": self._color_images[image_index, y, x],
                            "depth": self._depth_images[image_index, y, x],
                            "camera_position": self._positions[image_index]
                             }
        input_data["pixel"].shape = [batch_size, 2]
        input_data["color"].shape = [batch_size, 3]
        input_data["depth"].shape = [batch_size]
        input_data["camera_position"].shape = [4, 4]
        :return:
        """

        self.load_optimizer_state()
        self.optimizer.zero_grad()

        losses, data_batch = self.sample_and_backward_batch(state, state.frame.get_pixel_probs(), model)
        if is_image_active_sampling:
            new_pixel_weights = self._image_active_sampler.estimate_pixels_weights(
                data_batch['pixel'],
                losses['loss'],
                state.frame.get_pixel_probs())

            losses, data_batch = self.sample_and_backward_batch(state, new_pixel_weights, model)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.save_optimizer_state()

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

    def save_optimizer_state(self):
        self.opt_params = self.optimizer.state_dict()

    def load_optimizer_state(self):
        if self.opt_params is not None:
            self.optimizer.load_state_dict(self.opt_params)

    @staticmethod
    def forward_model(model, data_batch):
        output = model.forward(data_batch["pixel"], data_batch['camera_position'])
        return model.losses(output, data_batch['color'], data_batch['depth'])

    def reset_params(self):
        self.opt_params = None

    @staticmethod
    def backward_mean_loss(model, losses):
        mean_loss = model.mean_loss(losses['loss'])
        mean_loss.backward()
        return mean_loss.item()

    @staticmethod
    def send_batch_to_model_device(batch):
        batch['pixel'] = batch['pixel'].cuda()
        batch['color'] = batch['color'].cuda()
        batch['depth'] = batch['depth'].cuda()
        batch['camera_position'] = batch['camera_position'].cuda()

    @staticmethod
    def log_losses(writer, loss, i, verbose):
        if verbose:
            writer.add_scalar('image/coarse_image_loss', torch.mean(loss['coarse_image_loss']).item(), i,
                              new_style=True)
            writer.add_scalar('image/fine_image_loss', torch.mean(loss['fine_image_loss']).item(), i,
                              new_style=True)
            writer.add_scalar('depth/coarse_depth_loss', torch.mean(loss['coarse_depth_loss']).item(), i,
                              new_style=True)
            writer.add_scalar('depth/fine_depth_loss', torch.mean(loss['fine_depth_loss']).item(), i,
                              new_style=True)
            writer.add_scalar('loss', torch.mean(loss['loss']).item(), i,
                              new_style=True)

