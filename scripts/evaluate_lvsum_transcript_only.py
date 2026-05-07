#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
This script evaluates the LVSum dataset purely from transcriptions. No visuals
"""
import argparse
import json
import os
import sys
import time

import av
import anthropic
from openai import OpenAI
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer

from utils.eval_utils import parse_summary_response, compute_metrics
from utils.model_utils import GeminiModelSingleton
from utils.data_utils import create_hash

video_summary_prompt_template_orig = """
You are an expert video summarization assistant.
Your task is to read timestamped video transcriptions and select only the moments that are essential, important, or representative of the video's core content.
You must identify which timestamps should be included in the summary.

A timestamp is "summary-worthy" if it meets ANY of these:
- Introduces a key topic or event
- Shows a major action, decision, or outcome
- Conveys important factual information
- Includes a meaningful scene change
- Reflects emotionally significant or narratively important moments
- Is relevant for a short video highlight

Do NOT include timestamps that are repetitive, filler, disfluent, or off-topic.
Return only the selected timestamps, a relevance score from 0 to 3 (where 0 means least important and 3 means most important), and corresponding short summary of contents.

Here are the timestamped transcriptions:
{transcriptions}

Format your output as a list, for example:

Segment 1: 00:00:10 - 00:00:40 | Score: 3 | Description: Introduction to the main topic.
Segment 2: 00:05:20 - 00:05:50 | Score: 2 | Description: Explanation of key concept.
Segment 3: 00:12:00 - 00:12:30 | Score: 1 | Description: Additional example provided.
"""

video_summary_prompt_template = """
You are given a video transcript. Please provide a concise summary consisting of multiple important segments extracted from the video transcripts. For each segment, include:

The start and end timestamps in the format HH:MM:SS - HH:MM:SS.
A relevance score from 0 to 3, where 0 means least important and 3 means most important.
A brief textual description of the segment content.
Current video length is {video_length} seconds. The combined duration of all summarized segments should not exceed 15% of the total length of the video.

Transcript:
{transcriptions}

Format your output as a list, for example:

Segment 1: 00:00:10 - 00:00:40 | Score: 3 | Description: Introduction to the main topic.
Segment 2: 00:05:20 - 00:05:50 | Score: 2 | Description: Explanation of key concept.
Segment 3: 00:12:00 - 00:12:30 | Score: 1 | Description: Additional example provided.
"""


def get_gemini_summary(model, prompt, response_cache_dir="/mnt/task_wrapper/user_output/artifacts/response_cache_dir",
                       model_name="gemini-2.5-pro"):
    os.makedirs(response_cache_dir, exist_ok=True)
    generation_config = {
        "temperature": 0.0,
    }
    hash_code = create_hash(prompt, model_name)
    cache_file_path = os.path.join(response_cache_dir, hash_code + ".txt")

    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as cache_file:
            summary = cache_file.read()
    else:
        try:
            response = model.generate_content(prompt,
                                              generation_config=generation_config)
            summary = response.text

            with open(cache_file_path, 'w') as cache_file:
                cache_file.write(summary)
            time.sleep(5)
        except Exception as e:
            print(e)
            summary = ""

    return summary


def get_anthropic_summary(model, prompt, model_name,
                          response_cache_dir="/mnt/task_wrapper/user_output/artifacts/response_cache_dir", ):
    os.makedirs(response_cache_dir, exist_ok=True)

    # Map internal model names to public Anthropic model IDs
    if model_name == "sonnet-4.5":
        api_model_name = "claude-sonnet-4-20250514"
    elif model_name == "opus-4.5":
        api_model_name = "claude-opus-4-20250514"
    else:
        raise ValueError("Invalid model name")

    hash_code = create_hash(prompt, model_name) + ".txt"
    response_cache_path = os.path.join(response_cache_dir, hash_code)

    if os.path.exists(response_cache_path):
        with open(response_cache_path, 'r') as response_cache_file:
            response_text = response_cache_file.read()
            return response_text

    # Use native Anthropic API
    response = model.messages.create(
        model=api_model_name,
        max_tokens=4096,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_text = response.content[0].text
    with open(response_cache_path, 'w') as response_cache_file:
        response_cache_file.write(response_text)

    return response_text


def get_qwen3vl_summary(model, processor, prompt,
                        response_cache_dir="/mnt/task_wrapper/user_output/artifacts/response_cache_dir",
                        model_name="qwen3vl"):
    os.makedirs(response_cache_dir, exist_ok=True)
    hash_code = create_hash(prompt, model_name)
    cache_file_path = os.path.join(response_cache_dir, hash_code + ".txt")

    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as cache_file:
            summary = cache_file.read()
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)

        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=2048,
                                       temperature=0.0, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        summary = output_text[0]
        with open(cache_file_path, 'w') as cache_file:
            cache_file.write(summary)

    return summary


def get_sglang_summary(model, prompt, model_name,
                       response_cache_dir="/mnt/task_wrapper/user_output/artifacts/response_cache_dir"):
    os.makedirs(response_cache_dir, exist_ok=True)
    hash_code = create_hash(prompt, model_name) + ".txt"
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
                        "text": prompt
                    },
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


def main(args):
    if args.model == "gemini":
        model = GeminiModelSingleton.get_instance("gemini-2.5-pro")
        model_name = "gemini-2.5-pro"
    elif args.model == 'sonnet-4.5' or args.model == 'opus-4.5':
        model_name = args.model
        # Use public Anthropic API
        model = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    elif args.model == 'qwen3vl-235b':
        model = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    test_file_path = args.input_file_path  # "data/ego4d_video_summarization/annotated_data/annotated_video_summarization_89_filtered_min_7_videos.json"
    transcript_dir = args.transcript_dir  # "/mnt/temp_fs_data/test_transcripts/"
    test_set_videos = set()
    with open(test_file_path, 'r') as test_file:
        test_dict_list = json.load(test_file)
        summary_predictions = list()
        for test_dict in test_dict_list:
            if test_dict['video_path'] in test_set_videos:
                continue
            test_set_videos.add(test_dict['video_path'])
            video_path = test_dict['video_path']
            with av.open(video_path) as container:
                video_length = container.duration / av.time_base

            video_id = video_path.split("/")[-1].split(".")[0]

            with av.open(video_path) as container:
                duration = container.duration / av.time_base

            transcript_file_name = video_path.split("/")[-1].split(".")[0]
            input_file_path = os.path.join(transcript_dir, transcript_file_name + ".txt")
            if not os.path.exists(input_file_path):
                sys.exit(1)

            with open(input_file_path, 'r') as input_file:
                transcripts = input_file.read()

            video_summary_prompt = video_summary_prompt_template.format(transcriptions=transcripts,
                                                                        video_length=video_length)

            if args.model == "gemini":
                summary = get_gemini_summary(model, video_summary_prompt)
            elif args.model == 'qwen3vl-235b':
                summary = get_sglang_summary(model, video_summary_prompt, model_name=args.model)
            elif args.model in ['opus-4.5', 'sonnet-4.5']:
                summary = get_anthropic_summary(model, video_summary_prompt, model_name=model_name)
            else:
                raise ValueError("Unknown model")
            print(summary)

            try:
                summary_segments = parse_summary_response(summary)
            except Exception as e:
                import pdb;
                pdb.set_trace()
                print("ERROR:", str(e))
                print("ERROR: JSON decoding error: " + summary)
                summary_segments = []

            prediction_json = {
                "video_path": video_path,
                "video_id": transcript_file_name,
                "video_length": duration,
                "summary_timestamps": summary_segments
            }
            print("===============================")
            summary_predictions.append(prediction_json)
            if len(summary_segments) % 5 == 0:
                with open(args.output_file_path, 'w') as output_file:
                    json.dump(summary_predictions, output_file, indent=4)

        with open(args.output_file_path, 'w') as output_file:
            json.dump(summary_predictions, output_file, indent=4)

        compute_metrics(args.output_file_path, args.input_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate applebot dataset')

    # general
    parser.add_argument('-i', '--input-file-path', help='test ground truth file', required=False, type=str,
                        default='annotations/lvsum_72_dataset.json')
    parser.add_argument('-o', '--output-file-path', help='prediction file', required=False, type=str)
    parser.add_argument('-m', '--model', help='gemini|qwen3vl-235b|sonnet-4.5|opus-4.5',
                        required=False, type=str)
    parser.add_argument('-t', '--transcript-dir', help='location of transcripts dir', required=False,
                        default="/mnt/temp_fs_data/test_transcripts/", type=str)

    args = parser.parse_args()

    main(args)
