class Frame:
    def __init__(self, image, depth, valid_pixels):
        assert len(image.shape) == 3
        assert len(depth.shape) == 2
        assert depth.shape == valid_pixels.shape
        self.color_image = image
        self.depth_image = depth
        self.valid_pixels = valid_pixels

