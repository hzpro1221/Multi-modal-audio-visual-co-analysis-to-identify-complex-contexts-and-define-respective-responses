import os
import json
import subprocess
import re

# === Config ===
OUTPUT_FOLDER = 'video'
FFMPEG_PATH = '/usr/bin/ffmpeg'

with open('dataset.json') as f:
    DATASET = json.load(f)

VIDEO_URLS = []
for sample in DATASET:
    if sample['url'] not in VIDEO_URLS:
        VIDEO_URLS.append(sample['url']) 

# === Patterns ===
CHAPTER_PATTERN = re.compile(r"^(?P<video_name>.+?) - (?P<chapter_id>\d{3}) (?P<chapter_name>.+?) \[(?P<video_id>[\w-]{11})\]\.mp4$")
VIDEO_PATTERN = re.compile(r"^(?P<video_name>.+?) \[(?P<video_id>[\w-]{11})\]\.mp4$")

# === Functions ===

def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_video_ids(urls):
    return [url[16:] for url in urls]  # assumes fixed prefix

def download_videos(urls):
    for url in urls:
        try:
            command = [
                'yt-dlp',
                '--ffmpeg-location', os.path.dirname(FFMPEG_PATH),
                '--split-chapters',
                '-f', 'bestvideo[height<=360]+bestaudio/best[height<=360]',
                '--merge-output-format', 'mp4',
                url
            ]
            subprocess.run(command, check=True)
            print(f"✅ Downloaded: {url}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {url} — {e}")

def collect_metadata():
    metadata = []
    origin_videos = {}

    for root, _, files in os.walk('.'):
        for file in files:
            chapter_match = CHAPTER_PATTERN.match(file)
            video_match = VIDEO_PATTERN.match(file)

            if chapter_match:
                metadata.append({
                    "video_name": chapter_match.group('video_name'),
                    "chapter_name": chapter_match.group('chapter_name'),
                    "video_path": None,
                    "chapter_path": os.path.abspath(os.path.join(root, file)),
                })
            elif video_match:
                origin_videos[video_match.group('video_name')] = os.path.abspath(os.path.join(root, file))

    # Attach video paths to metadata
    for item in metadata:
        video_name = item["video_name"]
        if video_name in origin_videos:
            item["video_path"] = origin_videos[video_name]

    return metadata

def save_metadata_to_json(metadata, filename='metadata.json'):
    with open(filename, 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"📁 Metadata saved to {filename}")

# === Main ===

def main():
    ensure_output_folder()
    download_videos(VIDEO_URLS)
    metadata = collect_metadata()
    save_metadata_to_json(metadata)

if __name__ == "__main__":
    main()