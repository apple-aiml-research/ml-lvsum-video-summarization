#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch
import os
from google import genai
import anthropic
import torchvision.transforms as T
from PIL import Image
import av
from torchvision.transforms.functional import InterpolationMode
from openai import OpenAI
from utils.conversions import frame_to_timestamp_HH_MM_SS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GeminiModelSingleton:
    _instance = None
    _model_name = None

    @classmethod
    def get_instance(cls, model_name="gemini-2.0-flash-exp"):
        if cls._instance is None:
            cls._client = genai.Client(
                api_key=os.environ.get('GEMINI_API_KEY', os.environ.get('GOOGLE_API_KEY'))
            )
            cls._model_name = model_name
            cls._instance = cls
            print("Gemini instance created")
        return cls._instance

    @classmethod
    def generate_content(cls, contents=None, generation_config=None, safety_settings=None, **kwargs):
        config = {}
        if generation_config:
            config.update(generation_config)
        return cls._client.models.generate_content(
            model=cls._model_name,
            contents=contents,
            config=config,
        )

class AnthropicClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = anthropic.Anthropic(
                api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
            print("Anthropic instance created")
        return cls._instance

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def _load_frames_av(video_path):
    """Returns (frames_dict, fps) where frames_dict maps frame_index -> np.ndarray (H,W,3)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = {}
    for i, frame in enumerate(container.decode(stream)):
        frames[i] = frame.to_ndarray(format="rgb24")
    container.close()
    return frames, fps


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    frames, fps = _load_frames_av(video_path)
    max_frame = len(frames) - 1
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(frames[frame_index]).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_video_with_timestamps(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    frames, fps = _load_frames_av(video_path)
    max_frame = len(frames) - 1
    pixel_values_list, num_patches_list, timestamps_list = [], [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(frames[frame_index]).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        timestamp = frame_to_timestamp_HH_MM_SS(frame_index, fps=fps)
        timestamps_list.append(timestamp)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, timestamps_list
