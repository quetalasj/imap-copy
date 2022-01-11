import torch
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self, model):
        # self._model = model.to(device)
        # self._optimizer = self._model.configure_optimizers()
        model.to('cuda')
        self.optimizer = model.configure_optimizers()
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

        data_batch['pixel'] = data_batch['pixel'].cuda()
        data_batch['color'] = data_batch['color'].cuda()
        data_batch['depth'] = data_batch['depth'].cuda()
        data_batch['camera_position'] = data_batch['camera_position'][None].repeat(data_batch['pixel'].shape[0], 1, 1)
        data_batch['camera_position'] = data_batch['camera_position'].cuda()
        self.optimizer.zero_grad()
        _, loss = model.loss(data_batch)    # TODO: separate forward & loss
        loss['loss'].backward()
        self.optimizer.step()
        self.opt_params = self.optimizer.state_dict()
        return loss['loss'].item()

    def reset_params(self):
        self.opt_params = None


class ClassicalModelTrainer:
    def __init__(self, model):
        # self._model = model.to(device)
        # self._optimizer = self._model.configure_optimizers()
        model.to('cuda')
        self.optimizer = model.configure_optimizers()

    def train(self, model, data_batch):
        """
        Train the model one epoch
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
        input_data["camera_position"].shape = [batch_size, 4, 4]
        standard batch_size=4096
        :return:
        """

        data_batch['pixel'] = data_batch['pixel'].cuda()
        data_batch['color'] = data_batch['color'].cuda()
        data_batch['depth'] = data_batch['depth'].cuda()
        data_batch['camera_position'] = data_batch['camera_position'].cuda()
        self.optimizer.zero_grad()
        _, loss = model.loss(data_batch)  # TODO: separate forward & loss
        loss['loss'].backward()
        self.optimizer.step()
        # print(f"loss: {loss['loss'].item()}")
        return loss['loss'].item()


class ModelTrainer2:
    def __init__(self, model):
        # self._model = model.to(device)
        # self._optimizer = self._model.configure_optimizers()
        model.to('cuda')
        self.optimizer = model.configure_optimizers()

    def train(self, model, data_batch):
        """
        Train the model one epoch
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
        input_data["camera_position"].shape = [batch_size, 4, 4]
        standard batch_size=4096
        :return:
        """

        data_batch['pixel'] = data_batch['pixel'].cuda()
        data_batch['color'] = data_batch['color'].cuda()
        data_batch['depth'] = data_batch['depth'].cuda()
        data_batch['camera_position'] = data_batch['camera_position'][None].repeat(data_batch['pixel'].shape[0], 1, 1)
        data_batch['camera_position'] = data_batch['camera_position'].cuda()
        self.optimizer.zero_grad()
        _, loss = model.loss(data_batch)  # TODO: separate forward & loss
        loss['loss'].backward()
        self.optimizer.step()
        # print(f"loss: {loss['loss'].item()}")
        return loss['loss'].item()