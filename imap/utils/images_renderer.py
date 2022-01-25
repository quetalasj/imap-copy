import torch
import numpy as np

from imap.utils.utils import (
    generate_image_meshgrid,
    get_ji_pixels_from_meshgrid,
    reshape_points_list_to_image,
    reshape_points_list_to_depth
)


class ImagesRenderer:
    def __init__(self, height, width, mean, std, scale=1, batch_size=4096):
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.scale = scale
        self.batch_size = batch_size
        self.fine_image = None
        self.coarse_image = None
        self.fine_depth = None
        self.coarse_depth = None
        self.course_depth_variance = None
        self.fine_depth_variance = None

    def _generate_image_frame(self):
        x, y = generate_image_meshgrid(self.height, self.width, self.scale)
        pixels = torch.from_numpy(get_ji_pixels_from_meshgrid(x, y, self.scale)).requires_grad_(False)
        return pixels

    def _yield_render(self, model, position):
        device = next(model.parameters()).device
        pixels = self._generate_image_frame().to(device)
        for i in range(pixels.shape[0] // self.batch_size):
            yield model(pixels[i * self.batch_size:i * self.batch_size + self.batch_size], position)

    @staticmethod
    def _check_mode(mode):
        assert mode == 'fine' or mode == 'coarse'

    def clear_output(self):
        del self.fine_image, self.coarse_image, self.fine_depth, self.coarse_depth, self.course_depth_variance ,


    def render_output(self, model, position):
        self.clear_output()
        for batch_output in self._yield_render(model, position):
            self.output.append(batch_output)

    def fine_image(self):
        return

    def get_image(self, model, position, mode):
        ImagesRenderer._check_mode(mode)

        for outpu in range(pixels.shape[0] // self.batch_size):
            output = model(pixels[i * self.batch_size:i * self.batch_size + self.batch_size], position)
            if mode == 'coarse':
                output_image.append(output[0])
            elif mode == 'fine':
                output_image.append(output[2])
        return reshape_points_list_to_image(output_image, self.height, self.width, self.scale)

    def render_depth(self, model, position, mode):
        ImagesRenderer._check_mode(mode)
        output_depth = []
        device = next(model.parameters()).device
        for i in range(pixels.shape[0] // self.batch_size):
            output = model(pixels[i * self.batch_size:i * self.batch_size + self.batch_size], position)
            if mode == 'coarse':
                output_depth.append(output[1])
            elif mode == 'fine':
                output_depth.append(output[3])
        return reshape_points_list_to_depth(output_depth, self.height, self.width, self.scale)

    def reconstruct_rendered_image(self, rendered_image):
        """
        Reconstruct rendered numpy image
        :param rendered_image:
        :return:
        """
        reconstructed_image = rendered_image * self.std[None, None] + self.mean[None, None]
        return np.clip(reconstructed_image / 255., 0, 1).astype(np.float32)

