from speech_to_text import speech_to_text
from pydub import AudioSegment

import os
import re
import subprocess
from main import content_retrieve
from text_to_speech import text_to_speech
import cv2
import numpy as np

SPLITTERS = {".", "!", "?"}
OUTPUT_CLIP_DIR = "output_clips"
os.makedirs(OUTPUT_CLIP_DIR, exist_ok=True)

def split_paragraph(paragraph: str) -> list[str]:
    pattern = f"[{''.join(re.escape(s) for s in SPLITTERS)}]"
    parts = re.split(f"({pattern})", paragraph)
    sentences = []
    current = ""
    for part in parts:
        if part in SPLITTERS:
            if current.strip():
                sentences.append(current.strip() + part)
                current = ""
        else:
            current += part
    if current.strip():
        sentences.append(current.strip())
    return sentences

def get_audio_duration(audio_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def loop_or_trim_video(video_path, duration, output_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    original_duration = float(result.stdout.strip())

    if original_duration >= duration:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart",
            output_path
        ]
    else:
        loop_count = int(duration // original_duration) + 1
        cmd = [
            "ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart",
            output_path
        ]
    subprocess.run(cmd, check=True)

def combine_audio_video_text(video_path, audio_path, text, output_path, idx=0):
    # Tạo đường dẫn text file tạm
    textfile_path = os.path.join(OUTPUT_CLIP_DIR, f"segment_{idx}_text.txt")

    # Gói dòng (wrap) văn bản mỗi 50 ký tự (có thể điều chỉnh)
    wrapped_text = "\n".join(re.findall(r'.{1,90}(?:\s+|$)', text))

    # Ghi văn bản xuống file
    with open(textfile_path, "w") as f:
        f.write(wrapped_text)

    drawtext_filter = (
        f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"textfile='{textfile_path}':"
        "x=20:y=H-th-80:fontsize=12:fontcolor=white:"
        "box=1:boxcolor=black@0.5:boxborderw=5:line_spacing=6"
    )

    # Draw segment index (top-left)
    drawtext_index = (
        f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"text='Segment {idx}':"
        "x=10:y=10:fontsize=16:fontcolor=yellow:"
        "box=1:boxcolor=black@0.5:boxborderw=5"
    )

    # Combine filters with a comma
    vf_filter = f"{drawtext_text},{drawtext_index}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-shortest",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", vf_filter,
        output_path
    ]
    subprocess.run(cmd, check=True)

def reencode_video(input_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)

def concat_videos_ffmpeg(video_paths, output_path):
    input_cmds = []
    filter_inputs = []
    for idx, path in enumerate(video_paths):
        input_cmds.extend(["-i", os.path.abspath(path)])
        filter_inputs.append(f"[{idx}:v:0][{idx}:a:0]")

    filter_complex = "".join(filter_inputs) + f"concat=n={len(video_paths)}:v=1:a=1[outv][outa]"

    cmd = [
        "ffmpeg", "-y",
        *input_cmds,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-vsync", "2",
        "-async", "1",
        output_path
    ]

    subprocess.run(cmd, check=True)
    print(f"[✅] Final concatenated video saved to {output_path}")

def timestamp_to_ms(timestamp):
    h, m, s_ms = timestamp.split(":")
    s, ms = s_ms.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

if __name__ == "__main__":
    audio_path = "input.wav"
    segments = speech_to_text(audio_path)
    audio = AudioSegment.from_file(audio_path)

    final_video_paths = []

    for idx, segment in enumerate(segments):
        try:
            start_ms = timestamp_to_ms(segment["timestamp"][0])
            end_ms = timestamp_to_ms(segment["timestamp"][1])
            
            segment_audio = audio[start_ms:end_ms]
            segment_audio_path = os.path.join(OUTPUT_CLIP_DIR, f"segment_{idx}.wav")
            segment_audio.export(segment_audio_path, format="wav")

            sentence = segment["text"]
            queried_video_path, queried_video_score = content_retrieve(sentence)
            video_tmp_path = os.path.join(OUTPUT_CLIP_DIR, f"video_base_{idx}.mp4")

            audio_duration = get_audio_duration(segment_audio_path)
            loop_or_trim_video(queried_video_path, duration=audio_duration, output_path=video_tmp_path)

            final_clip_path = os.path.join(OUTPUT_CLIP_DIR, f"sentence_{idx}.mp4")
            combine_audio_video_text(
                video_path=video_tmp_path,
                audio_path=segment_audio_path,
                text=sentence,
                output_path=final_clip_path
            )

            encoded_clip_path = os.path.join(OUTPUT_CLIP_DIR, f"encoded_{idx}.mp4")
            reencode_video(final_clip_path, encoded_clip_path)

            final_video_paths.append(encoded_clip_path)
        except:
            continue 

    output_path = "final_output.mp4"
    concat_videos_ffmpeg(final_video_paths, output_path)
