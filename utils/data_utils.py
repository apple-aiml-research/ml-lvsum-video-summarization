#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import hashlib
import io
import av
import numpy as np
from utils.conversions import frame_to_timestamp_HH_MM_SS
from torchvision import transforms as T

def sample_video_frames_evenly(video_path, num_frames=8, resolution=336):
    """
    Evenly sample `num_frames` frames from a video and return them as PIL Images.
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Total number of frames (may be None for some codecs)
    total_frames = video_stream.frames
    fps = float(video_stream.average_rate)
    # Fallback: if total_frames is missing, compute using duration × fps
    if total_frames is None:
        total_frames = int(video_stream.duration * video_stream.time_base * fps)

    resize = T.Resize((resolution, resolution))
    # Select indices to sample
    indices = [
        int(i * total_frames / num_frames + total_frames / (2 * num_frames))
        for i in range(num_frames)
    ]

    frames = []
    current_index = 0
    target_set = set(indices)
    timestamps = []
    # Decode sequentially, collect target frames
    for frame in container.decode(video=0):
        if current_index in target_set:
            img = frame.to_image().convert("RGB")
            img = resize(img)
            timestamp = frame_to_timestamp_HH_MM_SS(current_index, fps=fps)
            timestamps.append(timestamp)

            frames.append(img)
            if len(frames) == num_frames:
                break
        current_index += 1

    container.close()
    return frames, timestamps

def sample_frames_with_timestamps(frames, num_samples, fps):
    total_frames = len(frames)

    if total_frames < num_samples:
        raise ValueError("Not enough frames to sample from.")

    # Get evenly spaced indices
    indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)

    # Sample frames and compute timestamps
    sampled = [
        {
            "frame": frames[i],
            "index": i,
            "timestamp_sec": i / fps
        }
        for i in indices
    ]

    return sampled



def create_hash(*values):
    # Concatenate the string representations of the values
    concatenated = ''.join(map(str, values))
    # Create a hash object
    hash_object = hashlib.sha256(concatenated.encode())
    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()


def pil_to_bytes(img, format="JPEG"):
    """
    Convert a PIL.Image into raw image bytes (JPEG/PNG/etc.).
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=80)
    return buffer.getvalue()