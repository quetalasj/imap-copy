import numpy as np
import torch
import pytorch3d.transforms as transforms


class Frame:
    def __init__(self, image, depth):
        assert len(image.shape) == 3
        assert len(depth.shape) == 2
        self.color_image = image
        self.depth_image = depth


class State:
    def __init__(self, frame, position, requires_grad=False, ground_truth_position=None, region_weights=None,
                 positionweight=1, lr=1e-3, device="cuda"):
        self.frame = frame

        assert len(position.shape) == 2
        if ground_truth_position is not None:
            assert position.shape == ground_truth_position.shape

        self.ground_truth_position = ground_truth_position
        self.position = position.T  # TODO: for compatibility with legacy code; write code without .T
        # with torch.no_grad():
        #     # position9d = position_9d_from_matrix(torch.tensor(position, device=device)[None])[0]
        #     position9d = transforms.se3_log_map(torch.tensor(position, device=device, dtype=torch.float32)[None])[0]
        # self.position = position9d.clone().detach().requires_grad_(requires_grad)
        # self.optimizer = torch.optim.Adam(params=[self.position], lr=lr)
        # self.weight = weight
        # self.region_weights = region_weights if region_weights is not None else np.ones((8, 8))

    # @property
    # def matrix_position(self):
    #     return torch.transpose(transforms.se3_exp_map(self.position[None]), 1, 2)[0].cpu().detach().numpy()
        # return matrix_from_9d_position(self.position[None])[0].cpu().detach().numpy()