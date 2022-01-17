import torch


class ModelTrainer:
    def __init__(self, parameters,  image_active_sampler, lr=0.005, **kwargs):
        self.optimizer = torch.optim.Adam(parameters, lr=lr, **kwargs)
        self.opt_params = None
        self._image_active_sampler = image_active_sampler

    def train(self, model, state):
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
        y, x = self._image_active_sampler.sample_pixels(state.frame.get_pixel_probs())
        data_batch = self._image_active_sampler.get_training_data(state, y, x)

        if self.opt_params is not None:
            self.optimizer.load_state_dict(self.opt_params)

        ModelTrainer.send_batch_to_model_device(data_batch)

        self.optimizer.zero_grad()
        # first sampling
        output = model.forward(data_batch["pixel"], data_batch['camera_position'])
        losses = model.losses(output, data_batch['color'], data_batch['depth'])
        mean_loss = model.mean_loss(losses['loss'])
        mean_loss.backward()

        # second sampling
        new_pixel_weights = self._image_active_sampler.estimate_pixels_weights(data_batch['pixel'],
                                                           losses['loss'],
                                                           state.frame.get_pixel_probs())

        y, x = self._image_active_sampler.sample_pixels(new_pixel_weights)
        data_batch = self._image_active_sampler.get_training_data(state, y, x)
        ModelTrainer.send_batch_to_model_device(data_batch)
        
        output = model.forward(data_batch["pixel"], data_batch['camera_position'])
        losses = model.losses(output, data_batch['color'], data_batch['depth'])
        mean_loss = model.mean_loss(losses['loss'])
        mean_loss.backward()
        # optimization
        self.optimizer.step()
        self.opt_params = self.optimizer.state_dict()

        return losses

    def reset_params(self):
        self.opt_params = None

    @staticmethod
    def send_batch_to_model_device(batch):
        batch['pixel'] = batch['pixel'].cuda()
        batch['color'] = batch['color'].cuda()
        batch['depth'] = batch['depth'].cuda()
        batch['camera_position'] = batch['camera_position'].cuda()

