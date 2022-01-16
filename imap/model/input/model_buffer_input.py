from collections import deque
import numpy as np
import torch


class ModelBufferInput:
    def __init__(self,
                 points_per_frame,
                 image_active_sampler):
        """
        :param points_per_frame: number of (x,y) coordinate (points) to sample per image
        :param image_active_sampler:
        Provide data for the model
        """
        self._states = deque([])
        self._points_per_frame = points_per_frame
        self._image_active_sampler = image_active_sampler

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
        return self._prepare_frame(state)

    # noinspection PyTypeChecker
    def _prepare_frame(self, state):
        y, x = self._image_active_sampler.sample_pixels(self._points_per_frame,
                                                        state.frame.get_pixel_probs())
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

    def update_data(self, state):
        self._states.append(state)
