import torch
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self):
        # self._model = model.to(device)
        # self._optimizer = self._model.configure_optimizers()
        pass

    def train(self, model, input_data):
        """
        Train the model one epoch
        :param model:
        :param input_data: {
                            "pixel": np.array([x, y], dtype=np.float32),
                            "color": self._color_images[image_index, y, x],
                            "depth": self._depth_images[image_index, y, x],
                            "camera_position": self._positions[image_index]
                             }
        input_data["pixel"].shape = [batch_size, 2]
        input_data["color"].shape = [batch_size, 3]
        input_data["depth"].shape = [batch_size]
        input_data["camera_position"].shape = [batch_size, 4, 4]
        :return:
        """
        model.to('cuda')
        optimizer = model.configure_optimizers()

        for data in input_data:
            print(data['pixel'].shape)
            print(data['color'].shape)
            print(data['depth'].shape)
            print(data['camera_position'].shape)
            break
            data['pixel'] = data['pixel'].cuda()
            data['color'] = data['color'].cuda()
            data['depth'] = data['depth'].cuda()
            data['camera_position'] = data['camera_position'].cuda()
            optimizer.zero_grad()
            _, loss = model.loss(data) # TODO: separate forward & loss
            loss['loss'].backward()
            optimizer.step()
            print(f"loss: {loss['loss'].item()}")

