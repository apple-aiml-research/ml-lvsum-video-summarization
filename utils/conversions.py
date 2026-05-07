#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#


def convert_seconds_to_HH_MM_SS(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def frame_to_timestamp_HH_MM_SS(frame_index, fps=30.0):
    seconds = frame_index / fps
    return convert_seconds_to_HH_MM_SS(seconds)

def timestamp_HH_MM_SS_to_seconds(timestamp: str):
    parts = timestamp.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])