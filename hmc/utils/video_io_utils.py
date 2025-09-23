import torch
import imageio.v3 as iio

def get_video_fps(video_path):
    metadata = iio.immeta(video_path, exclude_applied=False)
    return int(metadata["fps"])

