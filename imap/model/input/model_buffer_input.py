from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelBufferInput:
    def __init__(self, points_per_frame, device="cuda"):
        """
        :param points_per_frame: number of (x,y) coordinate (points) to sample per image
        :param device:
        Provide data for the model
        """
        self._states = deque([])
        self._weights_region = deque([])
        self._points_per_frame = points_per_frame
        self._device = device

    def __len__(self):
        return len(self._states)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._states:
            raise StopIteration
        return self._prepare_batch()

    def _prepare_batch(self):
        state = self._states.popleft()
        weights_region = self._weights_region.popleft()
        return self._prepare_frame(state, weights_region)

    # noinspection PyTypeChecker
    def _prepare_frame(self, state, weights_region):
        if weights_region is not None:
            x, y = self.sample_pixels_with_region_weights(state.frame.color_image.shape,
                                                          weights_region)
            raise NotImplementedError
        else:
            x, y = self.sample_pixels(state.frame.color_image.shape)
        result = {
                "pixel": torch.from_numpy(np.array([x, y], dtype=np.float32).T),
                "color": torch.from_numpy(state.frame.color_image[y, x]),
                "depth": torch.from_numpy(state.frame.depth_image[y, x]),
                "camera_position": state.get_matrix_position()
                }
        assert len(result['pixel'].shape) == 2 and result['pixel'].shape[1] == 2
        assert len(result['color'].shape) == 2 and result['color'].shape[1] == 3
        assert len(result['depth'].shape) == 1
        assert len(result['camera_position'].shape) == 2 and result['camera_position'].shape[0] == 4 and \
               result['camera_position'].shape[1] == 4
        return result

    def sample_pixels(self, image_shape):
        x = np.random.randint(image_shape[1], size=self._points_per_frame)
        y = np.random.randint(image_shape[0], size=self._points_per_frame)
        return x, y

    def sample_pixels_with_region_weights(self, image_shape, region_weights):
        region_height = image_shape[0] // region_weights.shape[0]
        region_width = image_shape[1] // region_weights.shape[1]
        region_y, region_x = np.meshgrid(range(region_weights.shape[0]), range(region_weights.shape[1]))
        region_left = region_x.reshape(-1)
        region_bottom = region_y.reshape(-1)
        normalized_weights = region_weights.reshape(-1) / np.sum(region_weights)
        region_indices = np.random.choice(len(normalized_weights), size=self._points_per_frame,
                                          replace=True, p=normalized_weights)
        x = region_left[region_indices] * region_width + np.random.randint(
            region_width, size=self._points_per_frame)
        y = region_bottom[region_indices] * region_height + np.random.randint(
            region_height, size=self._points_per_frame)
        return x, y

    def update_data(self, state, weights_region=None):
        self._states.append(state)
        self._weights_region.append(weights_region)
        # self._color_images = self._camera_info.process_color_image(np.array([x.image for x in frames]))
        # self._depth_images = self._camera_info.process_depth_image(np.array([x.depth for x in frames]))