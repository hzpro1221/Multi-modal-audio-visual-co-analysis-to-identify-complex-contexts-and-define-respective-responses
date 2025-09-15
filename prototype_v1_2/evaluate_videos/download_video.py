import os
import json
import subprocess
import re

# === Config ===
OUTPUT_FOLDER = 'video'
FFMPEG_PATH = '/usr/bin/ffmpeg'
 
# Test video - ver1
# VIDEO_URLS = [
#     "https://youtu.be/mhDJNfV7hjk", "https://youtu.be/V85qV4cjC4A",
#     "https://youtu.be/WPKKp3hkVx0", "https://youtu.be/OmJ-4B-mS-Y",
#     "https://youtu.be/Kp2bYWRQylk", "https://youtu.be/kJOyhNL52M4",
#     "https://youtu.be/QYA7Jy8Z0lA", "https://youtu.be/42osCqiNaUQ",
#     "https://youtu.be/q-_ezD9Swz4", "https://youtu.be/1h2xzUo_hgA",
#     "https://youtu.be/mLivzfKJlu4", "https://youtu.be/qbDVMPR_vmI",
#     "https://youtu.be/e72WoGOhghE", "https://youtu.be/ATh5u_l3sJQ",
#     "https://youtu.be/fD-qF0cHAP4", "https://youtu.be/p-d5S9JHYQQ",
#     "https://youtu.be/_kGESn8ArrU", "https://youtu.be/qGDNNjGBc0U",
#     "https://youtu.be/TjPFZaMe2yw", "https://youtu.be/ktGG-Ehas80"
# ]

# Test video - ver2
VIDEO_URLS = [
    "https://youtu.be/J4RqCSD--Dg", "https://youtu.be/qYNweeDHiyU",
    "https://youtu.be/Quh6x4kG6VY", "https://youtu.be/lEfrr0Yr684",
    "https://youtu.be/b-yhKUINb7o", "https://youtu.be/ZXiruGOCn9s",
    "https://youtu.be/zeFt_JCA3b4", "https://youtu.be/rEDzUT3ymw4",
    "https://youtu.be/_aCuOwF1ZjU", "https://youtu.be/oJC8VIDSx_Q",
]

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