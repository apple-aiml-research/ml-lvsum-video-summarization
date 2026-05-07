#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
This script enables evaluation
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import av
import cv2
from openai import OpenAI

from utils.eval_utils import parse_summary_response, compute_metrics
from utils.model_utils import AnthropicClientSingleton, GeminiModelSingleton
from utils.conversions import convert_seconds_to_HH_MM_SS
from utils.data_utils import create_hash, sample_frames_with_timestamps, sample_video_frames_evenly
from data.prompts import instruction_prompt_template, post_prompt



RAW_VIDEOS_DIR = Path(__file__).parent.parent / "raw_videos"


def url_to_local_path(url: str) -> str:
    """Resolve a video URL to its local path in raw_videos/."""
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = parsed.path.strip("/").replace("/", "_") + ".mp4"
    return str(RAW_VIDEOS_DIR / name)


def evaluate_with_gemini(model, video_path,
                            transcript_dir="annotations/test_transcripts/",
                            response_cache_dir="./response_cache_dir/",
                            num_frames=96,
                            use_transcripts=True):
    os.makedirs(response_cache_dir, exist_ok=True)

    sampled_frames, timestamp_list = sample_video_frames_evenly(video_path, num_frames=num_frames, resolution=336)

    # For public genai API, use PIL images directly
    image_parts = sampled_frames

    timestamp_string = "These frames are sampled at " + ", ".join(timestamp_list)

    with av.open(video_path) as container:
        duration = container.duration / av.time_base

    video_id = video_path.split("/")[-1].split(".")[0]
    transcript_file_path = os.path.join(transcript_dir, video_id + ".txt")
    if use_transcripts is True:
        with open(transcript_file_path, 'r', encoding="utf-8",
                  errors="replace") as transcript_file:
            transcripts = transcript_file.read()
    else:
        transcripts = "[No transcript provided]"

    gemini_prompt = instruction_prompt_template.format(video_length=duration)
    user_prompt = f"Here are the frames that I will upload and transcripts. " + timestamp_string + "\nTranscript:\n\n" + \
                  transcripts + post_prompt

    generation_config = {
        "temperature": 0.0,
    }

    hash_code = create_hash(video_path, gemini_prompt, user_prompt, num_frames, "gemini") + ".txt"
    response_cache_path = os.path.join(response_cache_dir, hash_code)
    if os.path.exists(response_cache_path):
        print("From cache: ", video_path)
        with open(response_cache_path, 'r') as response_cache_file:
            description = response_cache_file.read()
            return description

    print("Recomputing.....")
    contents = [gemini_prompt, user_prompt] + image_parts
    try:
        response = model.generate_content(
            contents=contents,
            safety_settings={
                "HATE": "BLOCK_ONLY_HIGH",
                "HARASSMENT": "BLOCK_ONLY_HIGH",
                "DANGEROUS": "BLOCK_ONLY_HIGH",
                "SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH"
            },
            generation_config=generation_config
        )
        print(response.text)
        response_text = response.text
        with open(response_cache_path, "w") as response_cache_file:
            response_cache_file.write(response_text)
    except Exception as e:
        print(e)
        response_text = "{}"

    return response_text

def evaluate_with_qwen3vl_235b(model, video_path,
                                  transcript_dir="annotations/test_transcripts/",
                                  response_cache_dir="./response_cache_dir/",
                                  num_frames=96,
                                  use_transcripts=True):
    os.makedirs(response_cache_dir, exist_ok=True)

    video_file_name = video_path.split("/")[-1].split(".")[0]

    with av.open(video_path) as container:
        duration = container.duration / av.time_base

    transcript_file_path = os.path.join(transcript_dir, video_file_name + ".txt")

    if not os.path.exists(transcript_file_path):
        print("Could not find transcript file: " + transcript_file_path)
        sys.exit(1)

    if use_transcripts is True:
        with open(transcript_file_path, 'r', encoding="utf-8",
                  errors="replace") as transcript_file:
            transcripts = transcript_file.read()
    else:
        transcripts = "[No transcript provided]"

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (336, 336))
        _, buffer = cv2.imencode(".jpg", resized_frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")

    # sampled_frames = [base64Frames[i] for i in evenly_sample_indices(len(base64Frames), 64)]
    sampled = sample_frames_with_timestamps(base64Frames, num_samples=num_frames, fps=fps)

    sampled_frames = []
    timestamp_list = []  # "The frames are sampled at "
    for frame_data in sampled:
        frame = frame_data['frame']
        time_sec = frame_data['timestamp_sec']
        time_HH_MM_SS = convert_seconds_to_HH_MM_SS(time_sec)
        timestamp_list.append(time_HH_MM_SS)
        sampled_frames.append(frame)

    timestamp_string = "These frames are sampled at " + ", ".join(timestamp_list)

    # video_duration = convert_seconds_to_HH_MM_SS(video_length)
    # f" Total video duration is {video_duration}."
    qwen3vl_235b_prompt = instruction_prompt_template.format(video_length=duration)
    user_prompt = f"Here are the frames that I will upload and transcripts. " + timestamp_string + "\nTranscript:\n\n" + \
                  transcripts + post_prompt
    # user_prompt = "\nTranscript:\n\n" + transcripts + post_prompt

    hash_code = create_hash(sampled_frames, qwen3vl_235b_prompt, user_prompt, "qwen3vl-235b-full") + ".txt"
    response_cache_path = os.path.join(response_cache_dir, hash_code)

    if os.path.exists(response_cache_path):
        with open(response_cache_path, 'r') as response_cache_file:
            response_text = response_cache_file.read()
            return response_text

    response = model.chat.completions.create(
        model="Qwen/Qwen3-VL-235B-A22B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": qwen3vl_235b_prompt + user_prompt
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame}"
                            }
                        }
                        for frame in sampled_frames
                    ]
                ]
            }
        ],
        temperature=0.0
    )

    response_text = response.choices[0].message.content

    print(response_text)
    with open(response_cache_path, "w") as response_cache_file:
        response_cache_file.write(response_text)

    return response_text

def evaluate_with_anthropic(model, video_path,
                            transcript_dir="annotations/test_transcripts/",
                            response_cache_dir="./response_cache_dir/",
                            num_frames=96,
                            use_transcripts=True,
                            model_name="sonnet-4.5"
                            ):
    os.makedirs(response_cache_dir, exist_ok=True)

    # Map internal model names to public Anthropic model IDs
    if model_name == "sonnet-4.5":
        api_model_name = "claude-sonnet-4-20250514"
    elif model_name == "opus-4.5":
        api_model_name = "claude-opus-4-20250514"
    else:
        raise ValueError("Invalid model name")

    video_file_name = video_path.split("/")[-1].split(".")[0]

    with av.open(video_path) as container:
        duration = container.duration / av.time_base


    transcript_file_path = os.path.join(transcript_dir, video_file_name + ".txt")

    if not os.path.exists(transcript_file_path):
        print("Could not find transcript file: " + transcript_file_path)
        sys.exit(1)

    if use_transcripts is True:
        with open(transcript_file_path, 'r', encoding="utf-8",
                  errors="replace") as transcript_file:
            transcripts = transcript_file.read()
    else:
        transcripts = "[No transcript provided]"

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)


    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (336, 336))
        _, buffer = cv2.imencode(".jpg", resized_frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")

    sampled = sample_frames_with_timestamps(base64Frames, num_samples=num_frames, fps=fps)

    sampled_frames = []
    timestamp_list = []  # "The frames are sampled at "
    for frame_data in sampled:
        frame = frame_data['frame']
        time_sec = frame_data['timestamp_sec']
        time_HH_MM_SS = convert_seconds_to_HH_MM_SS(time_sec)
        timestamp_list.append(time_HH_MM_SS)
        sampled_frames.append(frame)

    timestamp_string = "These frames are sampled at " + ", ".join(timestamp_list)

    anthropic_prompt = instruction_prompt_template.format(video_length=duration)
    user_prompt = f"Here are the frames that I will upload and transcripts. " + timestamp_string + "\nTranscript:\n\n" + \
                  transcripts + post_prompt

    hash_code = create_hash(sampled_frames, anthropic_prompt, user_prompt, model_name) + ".txt"
    response_cache_path = os.path.join(response_cache_dir, hash_code)

    if os.path.exists(response_cache_path):
        with open(response_cache_path, 'r') as response_cache_file:
            response_text = response_cache_file.read()
            return response_text

    # Use native Anthropic API with vision
    content_blocks = [
        {
            "type": "text",
            "text": anthropic_prompt + user_prompt
        }
    ]

    # Add image blocks
    for frame in sampled_frames:
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame
            }
        })

    response = model.messages.create(
        model=api_model_name,
        max_tokens=4096,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": content_blocks
            }
        ]
    )

    response_text = response.content[0].text
    with open(response_cache_path, 'w') as response_cache_file:
        response_cache_file.write(response_text)

    return response_text


def evaluate(args):
    if args.output_file_path is None:
        # Default output to current directory if not specified
        output_file_path = args.model + "_lvsum_summary_predictions.json"
    else:
        output_file_path = args.output_file_path

    use_transcript = args.use_transcript
    if args.model == 'gemini':
        model = GeminiModelSingleton.get_instance("gemini-2.5-pro")
    elif args.model == 'sonnet-4.5' or args.model == 'opus-4.5':
        model_name = args.model
        model = AnthropicClientSingleton.get_instance()
    elif args.model == 'qwen3vl-235b':
        model = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")
    else:
        raise Exception("Please select the appropriate model")

    with open(args.input_file_path, 'r') as input_file:
        json_dict_list = json.load(input_file)

        video_path_set = set()
        summary_predictions = list()
        for json_dict in json_dict_list:
            video_url = json_dict['video_path']
            video_id = json_dict['video_id']

            if video_url in video_path_set:
                continue
            video_path_set.add(video_url)

            video_path = url_to_local_path(video_url)
            if not os.path.exists(video_path):
                print(f"Local video not found: {video_path}. Run scripts/download_videos.py first.")
                sys.exit(1)

            with av.open(video_path) as container:
                duration = container.duration / av.time_base

            if args.model == 'gemini':
                response = evaluate_with_gemini(model, video_path,
                                                   num_frames=int(args.num_frames),
                                                   transcript_dir=args.transcript_dir,
                                                   use_transcripts=use_transcript)

            elif args.model == 'qwen3vl-235b':
                response = evaluate_with_qwen3vl_235b(model, video_path,
                                                         num_frames=int(args.num_frames),
                                                         transcript_dir=args.transcript_dir,
                                                         use_transcripts=use_transcript,
                                                         )
            elif args.model in ['opus-4.5', 'sonnet-4.5']:
                response = evaluate_with_anthropic(model, video_path,
                                                   transcript_dir=args.transcript_dir,
                                                   use_transcripts=use_transcript,
                                                   num_frames=int(args.num_frames),
                                                   model_name=model_name)
            else:
                raise Exception("Please provide appropriate model for evaluation")

            try:
                summary_segments = parse_summary_response(response)
            except Exception as e:
                print("Exception while parsing summary response: ", str(e))
                print("ERROR: JSON decoding error: " + response)
                summary_segments = []

            prediction_json = {
                "video_path": video_path,
                "video_id": video_id,
                "video_length": duration,
                "summary_timestamps": summary_segments
            }

            summary_predictions.append(prediction_json)
            if len(summary_segments) % 5 == 0:
                with open(output_file_path, 'w') as output_file:
                    json.dump(summary_predictions, output_file, indent=4)

        with open(output_file_path, 'w') as output_file:
            json.dump(summary_predictions, output_file, indent=4)


def main(args):
    print("Transcripts will be considered: ", args.use_transcript)
    evaluate(args)
    compute_metrics(args.output_file_path, args.input_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LVSum dataset')

    # general
    parser.add_argument('-i', '--input-file-path', help='test ground truth file', required=False, type=str,
                        default='annotations/lvsum_72_dataset.json')
    parser.add_argument('-o', '--output-file-path', help='prediction file', required=False, type=str,
                        default=None)
    parser.add_argument('-m', '--model', help='gemini or opus-4.5', required=False, type=str)
    parser.add_argument('-p', '--model-checkpoint', help='model checkpoint',
                        required=False, type=str, default=None)

    parser.add_argument('-t', '--transcript-dir', help='transcript directory', required=False, type=str,
                        default='annotations/test_transcripts/')
    parser.add_argument('-f', '--num-frames', help='num frames', required=False, type=str,
                        default=64)
    parser.add_argument(
        "--use-transcript",
        action="store_true",
        help="use transcript while evaluating"
    )

    args = parser.parse_args()
    main(args)
