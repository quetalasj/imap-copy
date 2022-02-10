import torch
from tqdm.auto import tqdm, trange
from imap.trainers.train_logger import TrainLogger

#
# def prepare_model(train_model):
#     def wrapper(self, model, *args, **kwargs):
#


class ModelTrainer:
    def __init__(self, image_active_sampler, keyframe_set, device='cuda'):
        self.opt_params = None
        self._image_active_sampler = image_active_sampler
        self.localization_poses = []
        self._keyframe_set = keyframe_set
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
        with TrainLogger('model_training') as logger:
            optimizer_params = ModelTrainer.check_optimizer_params(optimizer_params)
            model.requires_grad_(True)
            model.cuda()
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
            for j, state in tqdm(enumerate(dataset_loader)):
                for i in trange(num_epochs):
                    optimizer.zero_grad()
                    state_loss = self.forward_model(model, optimizer, state, is_image_active_sampling)
                    state_loss.loss.backward()
                    optimizer.step()
                    # mean_loss = state_loss
                    sampled_keyframes = self._keyframe_set.sample_frames()
                    for keyframe_id, keyframe in sampled_keyframes:
                        optimizer.zero_grad()
                        sampled_frame_loss = self.forward_model(model, optimizer, keyframe.state,
                                                                is_image_active_sampling)
                        # mean_loss += sampled_frame_loss
                        sampled_frame_loss.loss.backward()
                        optimizer.step()
                        self._keyframe_set.update_loss(keyframe_id, sampled_frame_loss)

                    # mean_loss.divide(len(sampled_keyframes) + 1)

                    # optimizer.step()
                    logger.log_losses(state_loss, j * num_epochs + i, verbose=verbose)

                self._keyframe_set.add_keyframe(state, state_loss)
                # del sampled_frame_loss, sampled_keyframes, state_loss
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            # del loss, optimizer
        torch.cuda.empty_cache()

    def localization(self,
                     model,
                     tracking_dataset_loader,
                     num_epochs=100,
                     is_image_active_sampling=False,
                     optimizer_params=None,
                     verbose=True):
        with TrainLogger('localization') as logger:
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
                    loss = self.forward_model(model, optimizer, state, is_image_active_sampling)
                    logger.log_losses(loss, i, verbose=verbose)

                state.freeze_position()
                state._position.cpu()

                current_position = state.get_matrix_position().detach().numpy()
                self.localization_poses.append(current_position.copy())

        del state, loss, optimizer
        torch.cuda.empty_cache()
        return self.localization_poses

    def forward_model(self, model, optimizer, state, is_image_active_sampling):
        self.load_optimizer_state(optimizer)
        losses, data_batch = self.sample_and_forward_batch(state, state.frame.get_pixel_probs(), model)
        if is_image_active_sampling:
            new_pixel_weights = self._image_active_sampler.estimate_pixels_weights(
                data_batch['pixel'],
                losses.loss,
                state.frame.get_pixel_probs())

            losses, _ = self.sample_and_forward_batch(state, new_pixel_weights, model)

        self.save_optimizer_state(optimizer)
        return losses.mean_loss()

    def sample_and_forward_batch(self, state, pixel_weights, model):
        data_batch = self._image_active_sampler.sample_batch(state, pixel_weights, device=self._device)
        output = model.forward(data_batch["pixel"], data_batch['camera_position'])
        return model.losses(output, data_batch['color'], data_batch['depth']), data_batch

    def save_optimizer_state(self, optimizer):
        self.opt_params = optimizer.state_dict()

    def load_optimizer_state(self, optimizer):
        if self.opt_params is not None:
            optimizer.load_state_dict(self.opt_params)

    def reset_params(self):
        self.opt_params = None

    @staticmethod
    def check_optimizer_params(optimizer_params):
        if optimizer_params is None:
            optimizer_params = {'lr': 0.005}
        return optimizer_params
