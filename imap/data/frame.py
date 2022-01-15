class Frame:
    def __init__(self, image, depth):
        assert len(image.shape) == 3
        assert len(depth.shape) == 2
        self.color_image = image
        self.depth_image = depth
