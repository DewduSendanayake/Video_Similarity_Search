import numpy as np
from decord import VideoReader, cpu

def sample_frames(video_path, num_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return np.zeros((0, 224, 224, 3), dtype='uint8')
    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames = vr.get_batch(idxs).asnumpy()  # (N, H, W, 3) dtype likely uint8
    # if decord returns float, force uint8:
    if frames.dtype != np.uint8:
        frames = frames.astype('uint8')
    return frames
