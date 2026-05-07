#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
This script generates the transcriptions for LVSum dataset
"""
import json
import os

from openai import OpenAI
from pydub import AudioSegment

from utils.conversions import convert_seconds_to_HH_MM_SS

model = OpenAI()

test_file_path = "annotations/lvsum_72_dataset.json"
output_file_path = ""
test_set_videos = set()
audio_dir = "/mnt/temp_fs_data/test_audios/"
transcript_dir = "/mnt/temp_fs_data/test_transcripts/"
os.makedirs(transcript_dir, exist_ok=True)

with open(test_file_path, 'r') as test_file:
    test_dict_list = json.load(test_file)

    for test_dict in test_dict_list:
        if test_dict['video_path'] in test_set_videos:
            continue

        test_set_videos.add(test_dict['video_path'])
        video_path = test_dict['video_path']
        video_file_name = video_path.split('/')[-1].split('.')[0]

        audio_file_path = os.path.join(audio_dir, video_file_name + ".mp3")
        print("Generating transcriptions for " + audio_file_path)
        transcript_file_path = os.path.join(transcript_dir, video_file_name + ".txt")
        if not os.path.exists(transcript_file_path):

            size_bytes = os.path.getsize(audio_file_path)
            size_mb = size_bytes / (1024 * 1024)

            with open(audio_file_path, "rb") as f, open(transcript_file_path, 'w') as transcript_file:

                if size_mb > 25:
                    audio = AudioSegment.from_mp3(audio_file_path)
                    # Split into N millisecond chunks (~25MB ≈ 24 minutes at 16kHz mono)
                    chunk_length_ms = 20 * 60 * 1000  # 20 minutes
                    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

                    transcript_segments = []

                    for i, chunk in enumerate(chunks):
                        chunk_path = f"chunk_{i}.mp3"
                        chunk.export(chunk_path, format="mp3")

                        with open(chunk_path, "rb") as f:
                            transcript = model.audio.transcriptions.create(
                                model="whisper-1",
                                file=f,
                                response_format="verbose_json"
                            )
                            transcript_segments.extend(transcript.segments)

                        os.remove(chunk_path)

                else:
                    transcript = model.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json"
                    )
                    transcript_segments = transcript.segments
                transcripts = ""
                for segment in transcript_segments:
                    start = convert_seconds_to_HH_MM_SS(segment.start)
                    end = convert_seconds_to_HH_MM_SS(segment.end)
                    transcript_file.write(f"{start} - {end}: {segment.text}\n")
                    print(f"{start} - {end}: {segment.text}")
                    transcripts += f"{start} - {end}: {segment.text}\n"
        else:
            with open(transcript_file_path, 'r') as transcript_file:
                transcripts = transcript_file.read()
            print(transcripts)