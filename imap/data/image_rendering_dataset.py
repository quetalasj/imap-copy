import numpy as np
from torch.utils import data


class ImageRenderingDataset(data.Dataset):
    def __init__(self, color_images, depth_images, positions, camera_info):
        assert color_images.shape[:3] == depth_images.shape
        assert positions.shape == (color_images.shape[0], 4, 4)
        print(f"Read {color_images.shape} images array")

        self._camera_info = camera_info
        # camera_info.update_color_normalization_parameters(color_images)
        self._color_images = camera_info.process_color_image(color_images)
        self._depth_images = camera_info.process_depth_image(depth_images)
        self._positions = camera_info.process_positions(positions)
        print(f"Dataset size: {len(self)} pixels")
        self._num_points_per_frame = int(640 * 480 / 100)
        self._trained_points = 0
        self._iteration_index = 0

    def __len__(self):
        color_image_shape = self._color_images.shape
        return color_image_shape[0] * color_image_shape[1] * color_image_shape[2]

    def __getitem__(self, index):
        image_count, height, width = self._color_images.shape[:3]

        # image_index = index // (width * height)

        image_index = self._iteration_index
        self._trained_points += 1
        # print(self._trained_points)
        if self._trained_points >= self._num_points_per_frame:
            self._iteration_index += 1
            self._trained_points = 0
            if self._iteration_index >= image_count:
                self._iteration_index = 0

        # y = (index % (width * height)) // width
        # x = (index % (width * height)) % width
        x = np.random.randint(width)
        y = np.random.randint(height)
        return {
            "pixel": np.array([x, y], dtype=np.float32),
            "color": self._color_images[image_index, y, x],
            "depth": self._depth_images[image_index, y, x],
            "camera_position": self._positions[image_index]
        }

    def camera_info(self):
        return self._camera_info
