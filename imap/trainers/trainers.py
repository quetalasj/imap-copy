import torch


class ModelTrainer:
    def __init__(self, parameters, lr=0.005, **kwargs):
        self.optimizer = torch.optim.Adam(parameters, lr=lr, **kwargs)
        self.opt_params = None

    def train(self, model, data_batch):
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
        if self.opt_params is not None:
            self.optimizer.load_state_dict(self.opt_params)

        ModelTrainer.send_batch_to_model_device(data_batch)

        self.optimizer.zero_grad()
        output = model.forward(data_batch["pixel"], data_batch['camera_position'])
        losses = model.losses(output, data_batch['color'], data_batch['depth'])
        mean_loss = model.mean_loss(losses['loss'])
        mean_loss.backward()
        self.optimizer.step()
        self.opt_params = self.optimizer.state_dict()

        return_losses = {
            "coarse_image_loss": model.mean_loss(losses['coarse_image_loss']).item(),
            "coarse_depth_loss": model.mean_loss(losses['coarse_depth_loss']).item(),
            "fine_image_loss": model.mean_loss(losses['fine_image_loss']).item(),
            "fine_depth_loss": model.mean_loss(losses['fine_depth_loss']).item(),
            "loss": mean_loss.item()
        }
        return return_losses

    def reset_params(self):
        self.opt_params = None

    @staticmethod
    def send_batch_to_model_device(batch):
        batch['pixel'] = batch['pixel'].cuda()
        batch['color'] = batch['color'].cuda()
        batch['depth'] = batch['depth'].cuda()
        batch['camera_position'] = batch['camera_position'].cuda()

