#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import re
import json
import math
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from utils.conversions import timestamp_HH_MM_SS_to_seconds
from scipy.stats import kendalltau, spearmanr

def intervals_to_indicator(intervals: List[Tuple[int, int]], scores: List[float], video_len: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """Converts list of (start, end) intervals to binary indicator array of length video_len."""
    indicator = np.zeros(video_len, dtype=int)
    indicator1 = np.zeros(video_len, dtype=int)
    idx = 0
    for start, end in intervals:
        indicator[int(start):int(end)] = 1  # assumes end is exclusive
        indicator1[int(start):int(end)] = scores[idx]
        idx += 1

    return indicator, indicator1

def compute_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    if true_positive == 0:
        return 0.0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_overlap_from_indicators(
        preds: List[Tuple[int, int]],
        pred_scores: List[float],
        gts: List[Tuple[int, int]],
        gt_scores: List[float],
        video_len: int
) -> Tuple[float, int, int, float, float, float, float]:
    """
    Converts intervals to indicator arrays, then computes:
    - overlap count (1s in both)
    - total prediction duration
    - total ground truth duration
    - IoU
    - F1
    - kendall
    - spearman
    """
    pred_arr, pred_arr1 = intervals_to_indicator(preds, pred_scores, video_len)
    gt_arr, gt_arr1 = intervals_to_indicator(gts, gt_scores, video_len)

    if len(pred_arr) != len(gt_arr):
        raise ValueError(f"len(pred_arr) must match with len(gt_arr)")

    f1 = compute_f1(gt_arr, pred_arr)
    kendall_tau = kendalltau(gt_arr1, pred_arr1).correlation
    spearman_coeff = spearmanr(gt_arr1, pred_arr1).correlation

    overlap = np.sum(pred_arr & gt_arr)
    pred_total = np.sum(pred_arr)
    gt_total = np.sum(gt_arr)
    union = pred_total + gt_total - overlap
    iou = overlap / union if union > 0 else 0.0

    return overlap, pred_total, gt_total, iou, f1, kendall_tau, spearman_coeff

def compute_metrics(prediction_file_path, gt_file_path, ignore_missing_keys=False, l_0=0.15,
                    output_per_video_metrics=False) -> Any:

    with open(prediction_file_path, 'r') as prediction_file, open(gt_file_path, 'r') as gt_file:
        prediction_dict_list = json.load(prediction_file)
        video_id_to_prediction_map = dict()
        error = 0
        predicted_video_ids = set()
        average_predicted_summary_length = 0
        for prediction_dict in prediction_dict_list:
            prediction_video_id = prediction_dict['video_id'].split(".")[0]
            if prediction_video_id in predicted_video_ids:
                continue
            predicted_video_ids.add(prediction_video_id)
            predicted_annotations = prediction_dict['summary_timestamps']
            if len(predicted_annotations) == 0 and ignore_missing_keys is True:
                error += 1
                continue
            video_length = int(math.ceil(prediction_dict['video_length']))
            predicted_intervals = []
            predicted_scores = []
            predicted_summary_length = 0
            for predicted_annotation in predicted_annotations:
                start_time = predicted_annotation['start_time']
                end_time = predicted_annotation['end_time']
                if isinstance(start_time, str) and ':' in start_time:
                    start_time = timestamp_HH_MM_SS_to_seconds(start_time)
                else:
                    start_time = int(math.floor(start_time))
                if isinstance(end_time, str) and ':' in end_time:
                    end_time = timestamp_HH_MM_SS_to_seconds(end_time)
                else:
                    end_time = int(math.ceil(end_time))

                if start_time > video_length:
                    continue
                if end_time > video_length:
                    end_time = video_length

                predicted_intervals.append((start_time, end_time))
                predicted_scores.append(predicted_annotation['score'])
                predicted_summary_length += (end_time - start_time)
            video_id_to_prediction_map[
                prediction_video_id] = predicted_intervals, predicted_scores, predicted_summary_length, \
                float(prediction_dict['video_length'])
            average_predicted_summary_length += (predicted_summary_length / video_length)
        gt_json_dict_list = json.load(gt_file)
        print("####Predicted file size: ", len(video_id_to_prediction_map))
        print("####GT file size: ", len(gt_json_dict_list))
        video_id_to_gt_map = dict()
        for gt_json_dict in gt_json_dict_list:
            gt_video_id = gt_json_dict['video_id'].split(".")[0]
            if gt_video_id not in video_id_to_gt_map:
                video_id_to_gt_map[gt_video_id] = list()
            summary_timestamps = gt_json_dict['summary_timestamps']
            video_id_to_gt_map[gt_video_id].append(summary_timestamps)

        overall_avg_f1 = 0.0
        overall_iou = 0.0
        overall_avg_f1_with_penalty = 0.0
        overall_avg_kendall = 0.0
        overall_avg_spearman = 0.0
        overall_avg_kendall_with_penalty = 0.0
        overall_avg_spearman_with_penalty = 0.0
        overall_best_f1 = 0
        overall_best_f1_with_penalty = 0.0
        processed = 0
        per_video_metrics = dict()
        for video_id in video_id_to_gt_map:

            print("VideoId: ", video_id)
            annotations = video_id_to_gt_map[video_id]

            if video_id  not in video_id_to_prediction_map:
                raise ValueError("video id is not found in video_id_to_prediction_map")
            predicted_intervals, predicted_scores, pred_summ_len, video_length = video_id_to_prediction_map[video_id]
            penalty = 1
            l = pred_summ_len / video_length
            alpha = 4.6
            if l > l_0:
                penalty = math.exp(-1 * alpha * (l - l_0))

            print("Penalty: ", penalty)

            average_f1 = 0.0
            average_kendall = 0.0
            average_spearman = 0.0
            average_iou = 0.0
            ann_kendall_count = 0
            ann_spearman_count = 0
            best_f1 = 0.0

            for gt_intervals_list in annotations:
                gt_intervals = []
                gt_scores = []
                for annotation in gt_intervals_list:

                    if 'start_time' in annotation and 'end_time' in annotation:
                        start_time = annotation['start_time'] / 1000.0
                        end_time = annotation['end_time'] / 1000.0
                    else:
                        start_time = annotation['start_timestamp']
                        end_time = annotation['end_timestamp']

                    if isinstance(start_time, str) and ':' in start_time:
                        start_time = timestamp_HH_MM_SS_to_seconds(start_time)
                    else:
                        start_time = int(math.floor(start_time))
                    if isinstance(end_time, str) and ':' in end_time:
                        end_time = timestamp_HH_MM_SS_to_seconds(end_time)
                    else:
                        end_time = int(math.ceil(end_time))

                    gt_intervals.append((start_time, end_time))
                    gt_scores.append(annotation['score'])

                overlap, pred_total, gt_total, iou, f1, kendall_tau, spearman_coeff = compute_overlap_from_indicators(
                    preds=predicted_intervals,
                    pred_scores=predicted_scores,
                    gts=gt_intervals,
                    gt_scores=gt_scores,
                    video_len=math.ceil(video_length))
                best_f1 = max(best_f1, f1)

                average_f1 += f1
                average_iou += iou

                if not math.isnan(kendall_tau):
                    average_kendall += kendall_tau
                else:
                    average_kendall += 0
                ann_kendall_count += 1
                if not math.isnan(spearman_coeff):
                    average_spearman += spearman_coeff
                else:
                    average_spearman += 0
                ann_spearman_count += 1

                print("Overlap: ", overlap)
                print("pred_total: ", pred_total)
                print("gt_total: ", gt_total)
                print("iou: ", iou)
                print("f1-score: ", f1)
                print("best f1-score: ", best_f1)
                print("kendall tau: ", kendall_tau)
                print("spearman coeff: ", spearman_coeff)


            average_f1 = average_f1 / len(annotations)
            average_f1_with_penalty = average_f1 * penalty
            best_f1_with_penalty = best_f1 * penalty
            average_iou = average_iou / len(annotations)

            if ann_kendall_count > 0:
                average_kendall = average_kendall / ann_kendall_count
                average_kendall_with_penalty = average_kendall * penalty
            else:
                average_kendall = 0
                average_kendall_with_penalty = 0
            if ann_spearman_count > 0:
                average_spearman = average_spearman / ann_spearman_count
                average_spearman_with_penalty = average_spearman * penalty
            else:
                average_spearman = 0
                average_spearman_with_penalty = 0
            print("Avg: IoU for {} = {}".format(video_id, average_iou))
            print("Avg: f1-score for {} = {}".format(video_id, average_f1))
            print("Avg: f1-score with penalty for {} = {}".format(video_id, average_f1_with_penalty))

            print("Avg: Best f1-score for {} = {}".format(video_id, best_f1))
            print("Avg: Best f1-score with penalty for {} = {}".format(video_id, best_f1_with_penalty))

            print("Avg: kendall-tau: {} = {}".format(video_id, average_kendall))
            print("Avg: spearman-co-eff: {} = {}".format(video_id, average_spearman))
            print("Avg: kendall-tau with penalty: {} = {}".format(video_id, average_kendall_with_penalty))
            print("Avg: spearman-co-eff with penalty: {} = {}".format(video_id, average_spearman_with_penalty))
            print("=================================================")

            per_video_metrics[video_id] = (average_iou, average_f1, average_f1_with_penalty, best_f1,
                                           best_f1_with_penalty, average_kendall, average_kendall_with_penalty,
                                           average_spearman, average_spearman_with_penalty, l)

            overall_avg_f1 += average_f1
            overall_iou += average_iou
            overall_avg_f1_with_penalty += average_f1_with_penalty
            overall_avg_kendall += average_kendall
            overall_avg_spearman += average_spearman
            overall_avg_kendall_with_penalty += average_kendall_with_penalty
            overall_avg_spearman_with_penalty += average_spearman_with_penalty
            overall_best_f1 += best_f1
            overall_best_f1_with_penalty += best_f1 * penalty
            processed += 1

        overall_iou = overall_iou / len(video_id_to_gt_map)
        print("Overall IoU: ", overall_iou)
        overall_f1 = overall_avg_f1 / len(video_id_to_gt_map)
        print("Overall f1: ", overall_f1)
        overall_avg_f1_with_penalty = overall_avg_f1_with_penalty / len(video_id_to_gt_map)
        print("Overall f1 with penalty: ", overall_avg_f1_with_penalty)
        overall_best_f1 = overall_best_f1 / len(video_id_to_gt_map)
        print("Overall best f1: ", overall_best_f1)
        overall_best_f1_with_penalty = overall_best_f1_with_penalty / len(video_id_to_gt_map)
        print("Overall best f1 with penalty: ", overall_best_f1_with_penalty)

        overall_kendall_tau = overall_avg_kendall / len(video_id_to_gt_map)
        print("Overall kendal tau:", overall_kendall_tau)
        overall_kendall_tau_with_penalty = overall_avg_kendall_with_penalty / len(video_id_to_gt_map)
        print("Overall kendall tau with penalty: ", overall_kendall_tau_with_penalty)

        overall_spearman_coeff = overall_avg_spearman / len(video_id_to_gt_map)
        print("Overall spearman coeff: ", overall_spearman_coeff)
        overall_spearman_coeff_with_penalty = overall_avg_spearman_with_penalty / len(video_id_to_gt_map)
        print("Overall spearman coeff with penalty: ", overall_spearman_coeff_with_penalty)
        print("Average predicted summary length: ", average_predicted_summary_length / len(video_id_to_prediction_map))
        print("Processed: ", processed)
        print("Error: ", error)
        if output_per_video_metrics:
            return per_video_metrics
        return overall_f1, overall_best_f1, overall_kendall_tau, overall_spearman_coeff
    print("Error!")
    return None

def parse_summary_response(text):
    pattern = re.compile(
        r"Segment \d+: (\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2}) \| Score: (\d) \| Description: (.+?)(?=\nSegment \d+:|\Z)",
        re.DOTALL
    )

    segments = []
    for match in pattern.finditer(text):
        start_time, end_time, score, description = match.groups()
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "score": int(score),
            "description": description.strip()
        })

    if len(segments) == 0:
        pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})\s*\|\s*Score:\s*(\d+)\s*\|\s*Description:\s*(.*)"
        )

        matches = pattern.findall(text)

        # Convert to structured data
        segments = [
            {"start_time": m[0], "end_time": m[1], "score": int(m[2]), "description": m[3]}
            for m in matches
        ]

    if len(segments) == 0:
        pattern = re.compile(
            r"Segment\s+\d+:\s+"
            r"(?P<start_time>\d{2}:\d{2}:\d{2})\s+-\s+"
            r"(?P<end_time>\d{2}:\d{2}:\d{2})\s+\|\s+"
            r"Score:\s+(?P<score>\d+)\s+\|\s+"
            r"(?P<description>.+)"
        )

        segments = [
            {
                "start_time": m.group("start_time"),
                "end_time": m.group("end_time"),
                "score": int(m.group("score")),
                "description": m.group("description").strip(),
            }
            for m in pattern.finditer(text)
        ]

    if len(segments) == 0:
        pattern = re.compile(
            r"Segment\s+\d+:\s+(?P<start_time>\d{2}:\d{2})\s*-\s*(?P<end_time>\d{2}:\d{2})\s*\|\s*Score:\s*(?P<score>\d+)\s*\|\s*Description:\s*(?P<description>.+)")

        segments = [
            {
                "start_time": m.group("start_time"),
                "end_time": m.group("end_time"),
                "score": int(m.group("score")),
                "description": m.group("description").strip(),
            }
            for m in pattern.finditer(text)
        ]

    if len(segments) == 0:
        pattern = re.compile(
            r"""
            (?:\*\*)?Segment\s+\d+:\s+
            (?P<start_time>\d{2}:\d{2}(?::\d{2})?)\s*-\s*
            (?P<end_time>\d{2}:\d{2}(?::\d{2})?)\s*\|\s*
            Score:\s*(?P<score>\d+)
            (?:\*\*)?                                  # optional closing **
            (?:\s*\|\s*Description:\s*|\s*\nDescription:\s*)
            (?P<description>.*?)
            (?=\n(?:\*\*)?Segment\s+\d+:|\Z)
            """,
            re.DOTALL | re.VERBOSE
        )

        segments = []
        for match in pattern.finditer(text):
            segments.append({
                "start_time": match.group("start_time"),
                "end_time": match.group("end_time"),
                "score": int(match.group("score")),
                "description": match.group("description").strip()
            })

    return segments


def compute_cr_and_mc_from_output(input_file_path):
        with open(input_file_path, "r") as input_file:
            json_dict = json.load(input_file)
            overall_mc = 0.0
            overall_cr = 0.0
            total = 0
            for video_id in json_dict:
                results = json_dict[video_id]

                avg_mc = 0
                avg_cr = 0
                for result in results:
                    cr_score = result["content_relevance"]['score']
                    mc_score = result["modality_coherence"]['score']
                    avg_mc += mc_score
                    avg_cr += cr_score

                avg_mc /= len(results)
                avg_cr /= len(results)

                overall_mc += avg_mc
                overall_cr += avg_cr
                total += 1

            print("overall_mc:", overall_mc / total)
            print("overall_cr:", overall_cr / total)
            print("Total: ", total)