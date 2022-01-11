import cv2
import numpy as np


class TUMDatasetLoader(object):
    def __init__(self, color_images, depth_images, positions):
        """
        :param color_image_paths:
        :param depth_image_paths:
        :param positions:
        :param factor: 5000.0 for the 16-bit PNG files
        OR: factor = 1 for the 32-bit float images in the ROS bag files
        """
        self._color_image = color_images
        self._depth_image = depth_images
        self._positions = positions

    def __getitem__(self, index):
        return self._color_image[index], self._depth_image[index], self._positions[index]

    def __len__(self):
        return len(self._color_image)
