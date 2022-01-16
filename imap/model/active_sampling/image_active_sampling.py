from imap.utils.utils import generate_image_meshgrid, get_ij_pixels_from_meshgrid
import numpy as np


class ImageActiveSampling:
    def __init__(self, image_width, image_height, grid_size=(8, 8)):
        self._grid_width = grid_size[0]
        self._grid_height = grid_size[1]
        self._image_width = image_width
        self._image_height = image_height
        x, y = generate_image_meshgrid(self._image_width, self._image_height, scale=1)
        self._sampling_pixels = get_ij_pixels_from_meshgrid(x, y)
        self._random_generator = np.random.default_rng()

    def sample_pixels(self, size, p):
        """
        :param size: num pixels to sample
        :param p: probabilities.shape = (self._image_width * self._image_height)
        :return: i(y), j(x) coordinates
        """
        pixels = self._random_generator.choice(self._sampling_pixels, size, p=p, axis=0, shuffle=False)
        return pixels[:, 0], pixels[:, 1]

    # def sample_pixels(self, image_shape):
    #     x = np.random.randint(image_shape[1], size=self._points_per_frame)
    #     y = np.random.randint(image_shape[0], size=self._points_per_frame)
    #     return x, y

    # def sample_pixels_with_region_weights(self, image_shape, region_weights):
    #     region_height = image_shape[0] // region_weights.shape[0]
    #     region_width = image_shape[1] // region_weights.shape[1]
    #     region_y, region_x = np.meshgrid(range(region_weights.shape[0]), range(region_weights.shape[1]))
    #     region_left = region_x.reshape(-1)
    #     region_bottom = region_y.reshape(-1)
    #     normalized_weights = region_weights.reshape(-1) / np.sum(region_weights)
    #     region_indices = np.random.choice(len(normalized_weights), size=self._points_per_frame,
    #                                       replace=True, p=normalized_weights)
    #     x = region_left[region_indices] * region_width + np.random.randint(
    #         region_width, size=self._points_per_frame)
    #     y = region_bottom[region_indices] * region_height + np.random.randint(
    #         region_height, size=self._points_per_frame)
    #     return x, y

