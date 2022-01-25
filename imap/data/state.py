import numpy as np
import torch
import pytorch3d.transforms as transforms

from .frame import Frame


class State:
    def __init__(self,
                 color_image,
                 depth_image,
                 pixel_weights,
                 position,
                 ground_truth_position=None):
        """
        :param color_image:
        :param depth_image:
        :param pixel_weights:
        :param position:    [[R 0],
                            [T 1]]
        :param ground_truth_position:   [[R 0],
                                        [T 1]]
        """
        self.frame = Frame(color_image, depth_image, pixel_weights)

        State.check_position(position)
        if ground_truth_position is not None:
            State.check_position(ground_truth_position)
        self._ground_truth_position = ground_truth_position
        # self.position = position.T  # TODO: solve .T compatibility with legacy code
        with torch.no_grad():
            position9d = transforms.se3_log_map(torch.tensor(position, device='cpu', dtype=torch.float32)[None])[0]
        self._position = position9d.clone().detach()
        self.freeze_position()

    @staticmethod
    def check_position(position):
        assert position.shape == (4, 4)
        assert np.all(position[:3, 3] == 0)

    def train_position(self):
        self._position.requires_grad_(requires_grad=True)

    def freeze_position(self):
        self._position.requires_grad_(requires_grad=False)

    def get_matrix_position(self):
        return transforms.se3_exp_map(self._position[None])[0]
