# LVSum: A Benchmark for Timestamp-Aware Long Video Summarization

This repository contains the code and benchmark for **LVSum**, a dataset for evaluating timestamp-aware video summarization models on long-form videos.

This software project accompanies the research paper: [LVSum: A Benchmark for Timestamp-Aware Long Video Summarization](https://arxiv.org/abs/2604.10024) (link to be updated).

## Overview

LVSum provides a comprehensive benchmark for evaluating video summarization models that can:
- Identify important segments in long videos
- Generate timestamp-based summaries with start/end times
- Assign relevance scores to video segments
- Utilize both visual and transcript information

## Dataset

The benchmark includes annotated video data with:
- **72 videos** with multiple human annotations
- Timestamp-based segment annotations (start time, end time, relevance score, description)
- Video transcripts for multimodal evaluation

Dataset files are located in the `annotations/` directory:
- `lvsum_72_dataset.json` - Human annotations for 72 videos

## Supported Models

The benchmark supports evaluation with the following state-of-the-art models:

### Multimodal (Vision + Text):
- **Gemini 2.0 Flash / 2.5 Pro** (Google AI)
- **Claude Sonnet 4.5** and **Claude Opus 4.5** (Anthropic)
- **Qwen3-VL-235B** (via local deployment)

### Text-only (Transcript-based):
- Same models as above, using only transcript information
- Use `evaluate_lvsum_transcript_only.py` for transcript-only evaluation

## Usage

### Downloading Videos

Download all 72 raw videos from the URLs listed in `annotations/lvsum_72_dataset.json`:

```bash
python scripts/download_videos.py
```

Videos are saved to `raw_videos/` with filenames derived from their URLs. Already-downloaded videos are skipped automatically. The script exits with a non-zero status if any downloads fail and prints the failed URLs.

### Generating Transcripts

Generate timestamped transcripts for the dataset using Gemini 2.5 Pro. The script expects audio files (`.mp3`) extracted from the videos.

**Prerequisites:**

1. Extract audio from each video (e.g., using `ffmpeg`):
   ```bash
   ffmpeg -i raw_videos/<video>.mp4 -vn -acodec libmp3lame extracted_audios/<video>.mp3
   ```
   
2. Edit the path variables at the top of `scripts/generate_lvsum_transcripts.py` to point to your audio and output directories:
   ```python
   audio_dir = "/path/to/extracted/test_audios/"
   transcript_dir = "/path/to/output/test_transcripts/"
   ```

Then run:

```bash
python scripts/generate_lvsum_transcripts.py
```

Transcripts are saved as `.txt` files in `transcript_dir`, one per video, with lines in the format:

```
HH:MM:SS - HH:MM:SS: <transcribed text>
```

Audio files larger than 25 MB are automatically split into 20-minute chunks and transcribed with adjusted timestamps. Already-transcribed videos are skipped.

### Environment Setup

Set up API keys for the models you want to use:

```bash
# For Anthropic models (Claude)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google models (Gemini)
export GEMINI_API_KEY="your-google-api-key"
# or
export GOOGLE_API_KEY="your-google-api-key"
```

### Multimodal Evaluation (Vision + Transcript)

Evaluate with **Gemini**:
```bash
python scripts/evaluate_lvsum.py \
  -m gemini \
  -i annotations/lvsum_72_dataset.json \
  -o results/gemini_predictions.json \
  -t /path/to/transcripts \
  -f 96 \
  --use-transcript
```

Evaluate with **Claude Opus 4.5**:
```bash
python scripts/evaluate_lvsum.py \
  -m opus-4.5 \
  -i annotations/lvsum_72_dataset.json \
  -o results/opus_predictions.json \
  -t /path/to/transcripts \
  -f 96 \
  --use-transcript
```

Evaluate with **Qwen3-VL-235B-A22B-Instruct**:
```bash
python scripts/evaluate_lvsum.py \
  -m qwen3vl-235b \
  -i annotations/lvsum_72_dataset.json \
  -o results/qwen3vl_predictions.json \
  -t /path/to/transcripts \
  -f 96 \
  --use-transcript
```

### Transcript-only Evaluation

For text-only evaluation (no visual frames):

```bash
python scripts/evaluate_lvsum_transcript_only.py \
  -m opus-4.5 \
  -i annotations/lvsum_72_dataset.json \
  -o results/opus_transcript_only.json \
  -t /path/to/transcripts
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-m, --model` | Model name: `gemini`, `sonnet-4.5`, `opus-4.5`, `qwen3vl-235b` | Required |
| `-i, --input-file-path` | Path to ground truth annotation file | `annotations/lvsum_72_dataset.json` |
| `-o, --output-file-path` | Path to save predictions | Auto-generated |
| `-t, --transcript-dir` | Directory containing video transcripts | `/mnt/temp_fs_data/test_transcripts/` |
| `-f, --num-frames` | Number of frames to sample from video | `64` |
| `--use-transcript` | Include transcript in evaluation (multimodal) | `False` |

## Evaluation Metrics

The benchmark evaluates models on:

- **F1 Score**: Overlap between predicted and ground truth segments
- **IoU (Intersection over Union)**: Temporal overlap metric
- **Kendall's Tau**: Correlation of segment importance scores
- **Spearman's Correlation**: Rank correlation of segment relevance

Metrics include penalty adjustments for summary length violations (summaries exceeding 15% of video length).

## Output Format

Predictions are saved in JSON format:
```json
[
  {
    "video_path": "/path/to/video.mp4",
    "video_id": "video_id",
    "video_length": 1234.56,
    "summary_timestamps": [
      {
        "start_time": 10.5,
        "end_time": 25.3,
        "score": 3,
        "description": "Important segment description"
      }
    ]
  }
]
```

## Citation

If you use LVSum in your research, please cite our paper:

```bibtex
@misc{patel2026lvsumbenchmarktimestampawarelong,
      title={LVSum: A Benchmark for Timestamp-Aware Long Video Summarization}, 
      author={Alkesh Patel and Melis Ozyildirim and Ying-Chang Cheng and Ganesh Nagarajan},
      year={2026},
      eprint={2604.10024},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.10024}, 
}
```

## License

Copyright (C) 2026 Apple Inc. All Rights Reserved.

This project is licensed under Apple's proprietary license. See the [LICENSE](LICENSE) file for details.

This software includes third-party dependencies. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for details.

Data: [CC-BY-NC-ND](LICENSE_DATA) [Deed](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before getting started.

## Acknowledgments

This work builds upon the capabilities of state-of-the-art multimodal models including Claude (Anthropic), Gemini (Google), and Qwen-VL.