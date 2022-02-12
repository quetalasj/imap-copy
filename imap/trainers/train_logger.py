from torch.utils.tensorboard import SummaryWriter
import torch


class TrainLogger:
    def __init__(self, comment=""):
        self._comment = comment

    def __enter__(self):
        self.writer = SummaryWriter(comment=self._comment)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        torch.cuda.empty_cache()

    def log_losses(self, loss, i, verbose):
        if verbose:
            self.writer.add_scalar('image/coarse_image_loss', loss.coarse_image_loss.item(), i, new_style=True)
            self.writer.add_scalar('image/fine_image_loss', loss.fine_image_loss.item(), i, new_style=True)
            self.writer.add_scalar('depth/coarse_depth_loss', loss.coarse_depth_loss.item(), i, new_style=True)
            self.writer.add_scalar('depth/fine_depth_loss', loss.fine_depth_loss.item(), i, new_style=True)
            self.writer.add_scalar('loss', loss.loss.item(), i, new_style=True)
