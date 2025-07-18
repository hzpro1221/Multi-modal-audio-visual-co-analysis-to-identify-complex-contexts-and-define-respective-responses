import os
import json
import subprocess

# Change the path to the folder where videos will be saved
output_folder = 'videos'
os.makedirs(output_folder, exist_ok=True)

# List of video URLs
video_urls = [
    "https://youtu.be/mhDJNfV7hjk",
    "https://youtu.be/V85qV4cjC4A",
    "https://youtu.be/WPKKp3hkVx0",
    "https://youtu.be/OmJ-4B-mS-Y",
    "https://youtu.be/Kp2bYWRQylk",
    "https://youtu.be/kJOyhNL52M4",  
    "https://youtu.be/QYA7Jy8Z0lA",
    "https://youtu.be/42osCqiNaUQ",
    "https://youtu.be/q-_ezD9Swz4",
    "https://youtu.be/1h2xzUo_hgA",
    "https://youtu.be/mLivzfKJlu4",
    "https://youtu.be/qbDVMPR_vmI",
    "https://youtu.be/e72WoGOhghE",
    "https://youtu.be/ATh5u_l3sJQ",
    "https://youtu.be/fD-qF0cHAP4",
    "https://youtu.be/p-d5S9JHYQQ",
    "https://youtu.be/_kGESn8ArrU",
    "https://youtu.be/qGDNNjGBc0U",
    "https://youtu.be/TjPFZaMe2yw",
    "https://youtu.be/ktGG-Ehas80"
]

# Path to ffmpeg.exe
ffmpeg_path = '/usr/bin/ffmpeg'

# Download each video and split into chapters
for url in video_urls:
    try:
        command = [
            'yt-dlp',
            '--ffmpeg-location', os.path.dirname(ffmpeg_path),
            '--split-chapters',
            '-o', os.path.join(output_folder, '%(title)s/%(chapter_number)s - %(chapter)s.%(ext)s'),
            url
        ]
        subprocess.run(command, check=True)
        print(f"Downloaded and processed: {url}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")

# Save metadata information to a JSON file
metadata = []

# Get list of video files in the output folder
for root, dirs, files in os.walk(output_folder):
    for file in files:
        if file.endswith('.mp4'):
            chapter_name = os.path.splitext(file)[0]
            chapter_path = os.path.join(root, file)
            metadata.append({
                "video_name": os.path.basename(root),
                "chapter_name": chapter_name,
                "chapter_path": chapter_path,
            })

# Save metadata to JSON file
metadata_file_path = os.path.join(output_folder, 'metadata.json')
with open(metadata_file_path, 'w') as json_file:
    json.dump(metadata, json_file, indent=4)

print(f"Metadata file has been saved at: {metadata_file_path}")
