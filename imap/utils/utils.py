import numpy as np


def generate_image_meshgrid(height, width, scale):
    """
    :param height:
    :param width:
    :param scale:
    :return:    x.shape = (width // scale, height // scale),
                y.shape = (width // scale, height // scale)
    """
    y, x = np.meshgrid(range(height // scale), range(width // scale))
    return x, y


def get_pixels_from_meshgrid(x_meshgrid, y_meshgrid, scale):
    """
    :param x_meshgrid:
    :param y_meshgrid:
    :param scale:
    :return pixels: shape = (height // scale * width // scale, 2), dtype = int32

    Example:
        height=480, width=640
        scale = 2:
            array([[  0,   0],
                   [  2,   0],
                   [  4,   0],
                   ...,
                   [634, 478],
                   [636, 478],
                   [638, 478]], dtype=int32)
        scale = 1:
            array([[  0,   0],
                   [  1,   0],
                   [  2,   0],
                   ...,
                   [637, 479],
                   [638, 479],
                   [639, 479]], dtype=int32)
    """
    return (np.array([x_meshgrid, y_meshgrid], dtype=np.int32).T * scale).reshape(-1, 2)


def get_image_values_in_pixels(image, pixels):
    return image[pixels[:, 1], pixels[:, 0]]


def get_depths_in_pixels(depth_image, pixels):
    return get_image_values_in_pixels(depth_image, pixels)


def get_colors_in_pixels(color_image, pixels):
    return get_image_values_in_pixels(color_image, pixels)
