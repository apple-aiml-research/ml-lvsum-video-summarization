#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Download raw videos from URLs listed in the lvsum_72_dataset.json annotation file.
Videos are saved to raw_videos/ with filenames derived from their URLs.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ANNOTATIONS_PATH = Path(__file__).parent.parent / "annotations" / "lvsum_72_dataset.json"
OUTPUT_DIR = Path(__file__).parent.parent / "raw_videos"


def url_to_filename(url: str) -> str:
    """Derive a local filename from a video URL."""
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        # Fallback: sanitize the full path
        name = parsed.path.strip("/").replace("/", "_") + ".mp4"
    return name


def download_video(url: str, dest: Path) -> bool:
    """Download a single video. Returns True on success."""
    if dest.exists():
        print(f"  [skip] already exists: {dest.name}")
        return True

    tmp = dest.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:.1f}%  ({downloaded // (1024*1024)} / {total // (1024*1024)} MB)", end="", flush=True)

        tmp.rename(dest)
        print(f"\r  done → {dest.name}{' ' * 20}")
        return True

    except Exception as exc:
        print(f"\r  ERROR: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    # Collect unique URLs while preserving first-seen order
    seen = set()
    urls = []
    for entry in data:
        url = entry["video_path"]
        if url not in seen:
            seen.add(url)
            urls.append(url)

    print(f"Found {len(urls)} unique video URLs. Saving to: {OUTPUT_DIR}\n")

    failed = []
    for i, url in enumerate(urls, 1):
        filename = url_to_filename(url)
        dest = OUTPUT_DIR / filename
        print(f"[{i}/{len(urls)}] {filename}")
        success = download_video(url, dest)
        if not success:
            failed.append(url)
        # Small delay between requests to be polite to servers
        if i < len(urls):
            time.sleep(0.5)

    print(f"\nDone. {len(urls) - len(failed)}/{len(urls)} videos downloaded successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for url in failed:
            print(f"  {url}")
        sys.exit(1)


if __name__ == "__main__":
    main()
