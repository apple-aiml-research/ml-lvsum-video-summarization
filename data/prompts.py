#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

instruction_prompt_template = """
You are given a video. Please provide a concise summary consisting of multiple important segments extracted from the video. For each segment, include:

The start and end timestamps in the format HH:MM:SS - HH:MM:SS.
A relevance score from 0 to 3, where 0 means least important and 3 means most important.
A brief textual description of the segment content.
"""

post_prompt = """
Generate a video summary using both the visual content in the video frames and the transcript (if provided).

Format your output as a list, for example:

Segment 1: 00:00:10 - 00:00:40 | Score: 3 | Description: Introduction to the main topic.
Segment 2: 00:05:20 - 00:05:50 | Score: 2 | Description: Explanation of key concept.
Segment 3: 00:12:00 - 00:12:30 | Score: 1 | Description: Additional example provided.
"""

length_constraint_prompt_template = """
Current video length is {video_length} seconds. The combined duration of all summarized segments should not exceed 15% of the total length of the video.
"""