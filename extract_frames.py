"""Extract BDD100K frames at the timestamps in the GPS metadata."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True, help="Directory containing the source videos for one split.")
    parser.add_argument("--json_dir", type=Path, required=True, help="Directory containing <video_id>.json GPS metadata.")
    parser.add_argument("--split_file", type=Path, required=True, help="Split list; the first whitespace-separated field of each line is a video ID.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Frames are written to <output_dir>/<video_id>/<index>.jpg.")
    parser.add_argument("--video_ext", default=".mov", help="Source video extension (default: .mov).")
    parser.add_argument("--width", type=int, default=384, help="Output width in pixels (default: 384).")
    parser.add_argument("--height", type=int, default=216, help="Output height in pixels (default: 216).")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive.")
    if not args.video_ext.startswith("."):
        args.video_ext = "." + args.video_ext
    if "/" in args.video_ext or "\\" in args.video_ext:
        parser.error("--video_ext must be a filename extension.")
    for directory in (args.video_dir, args.json_dir):
        if not directory.is_dir():
            parser.error(f"Directory not found: {directory}")
    if not args.split_file.is_file():
        parser.error(f"Split file not found: {args.split_file}")
    return args


def read_video_ids(split_file, video_ext):
    video_ids = []
    seen = set()
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        video_id = line.split()[0]
        if video_id.endswith(video_ext):
            video_id = video_id[:-len(video_ext)]
        if video_id in ("", ".", "..") or "/" in video_id or "\\" in video_id:
            raise ValueError(f"Expected a video ID or filename, got {video_id!r}")
        if video_id not in seen:
            video_ids.append(video_id)
            seen.add(video_id)
    if not video_ids:
        raise ValueError(f"No video IDs found in {split_file}")
    return video_ids


def extract_video(video_id, args, cv2):
    video_path = args.video_dir / f"{video_id}{args.video_ext}"
    json_path = args.json_dir / f"{video_id}.json"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    with json_path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    locations = data["locations"]
    if not locations:
        raise ValueError(f"No locations in {json_path}")
    timestamps = [float(loc["timestamp"]) - float(data["startTime"]) for loc in locations]
    if any(not math.isfinite(ts) or ts < 0 for ts in timestamps):
        raise ValueError(f"Invalid relative timestamp in {json_path}")
    cap = cv2.VideoCapture(str(video_path))
    written = 0
    failed = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        frame_dir = args.output_dir / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, timestamp in enumerate(timestamps):
            # BDD timestamps and startTime are in milliseconds.
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)
            success, frame = cap.read()
            if not success:
                print(f"WARNING: {video_id}: cannot decode index {index} at {timestamp:g} ms")
                failed += 1
                continue
            frame = cv2.resize(frame, (args.width, args.height))
            output_path = frame_dir / f"{index:02d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Cannot write frame: {output_path}")
            written += 1
    finally:
        cap.release()
    return written, failed


def main():
    args = parse_args()
    video_ids = read_video_ids(args.split_file, args.video_ext)
    import cv2

    total_written = total_failed_frames = failed_videos = 0
    for number, video_id in enumerate(video_ids, start=1):
        print(f"[{number}/{len(video_ids)}] {video_id}", flush=True)
        try:
            written, failed = extract_video(video_id, args, cv2)
            total_written += written
            total_failed_frames += failed
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
            print(f"ERROR: {video_id}: {error}")
            failed_videos += 1
    print(f"Finished: {total_written} frames written, {total_failed_frames} frames failed, "
          f"{failed_videos} videos failed. Output: {args.output_dir}")
    return int(bool(total_failed_frames or failed_videos))


if __name__ == "__main__":
    raise SystemExit(main())
