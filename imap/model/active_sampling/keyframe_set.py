from imap.model.active_sampling.keyframe import Keyframe


class KeyframeSet:
    def __init__(self, num_samples=3):
        self._id = 0
        self._keyframes = {}    # {id: keyframe}
        self._last_keyframe = None
        self.num_samples = num_samples

    def add_keyframe(self, state, losses):
        if self._last_keyframe is not None:
            self._keyframes[self._id] = self._last_keyframe
            self._id += 1
        self._last_keyframe = Keyframe(state, losses.detach())

    def sample_frames(self):
        sampled_frames = sorted(self._keyframes.items(), key=lambda item: item[1].loss)[: self.num_samples]
        if self._last_keyframe is not None:
            sampled_frames.append((self._id, self._last_keyframe))
        return sampled_frames

    def update_loss(self, keyframe_id, sampled_frame_loss):
        if keyframe_id == self._id:
            self._last_keyframe.loss = sampled_frame_loss.detach()
        else:
            self._keyframes[keyframe_id].loss = sampled_frame_loss.detach()
