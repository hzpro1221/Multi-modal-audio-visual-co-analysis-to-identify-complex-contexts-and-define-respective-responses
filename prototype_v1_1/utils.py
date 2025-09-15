import os
import subprocess
from typing import List
from pathlib import Path
import math
import cv2
from typing import Union

def fixed_interval_video_splitter(
    video_path: str,
    output_dir: str,
    interval_sec: float,
    output_format: str = "mp4",
    prefix: str = "chunk"
) -> List[Path]:
    """
    Splits a video into fixed-duration segments and saves them to the specified output directory.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the video chunks will be saved.
        interval_sec (float): Duration (in seconds) of each video chunk.
        output_format (str, optional): Format of the output video chunks (e.g., 'mp4', 'avi'). Default is 'mp4'.
        prefix (str, optional): Prefix used for naming the output video files. Default is 'chunk'.

    Returns:
        List[Path]: A list of paths to the generated video chunk files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get duration of the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    # Compute number of chunks
    num_chunks = math.ceil(duration / interval_sec)
    output_paths: List[Path] = []

    for i in range(num_chunks):
        start_time = i * interval_sec
        output_file = Path(output_dir) / f"{prefix}_{i:04d}.{output_format}"
        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(interval_sec),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_file)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        output_paths.append(output_file)

    return output_paths

def extract_audio_from_video(
    video_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    audio_format: str = "wav"
) -> Path:
    """
    Extracts audio from a video file using the FFmpeg command-line tool.

    Args:
        video_path (str | Path): Path to the input video file (e.g., .mp4, .mov).
        output_path (str | Path, optional): Path where the output audio will be saved.
                                            If None, uses the same name as the video with new extension.
        audio_format (str): Desired format of the extracted audio (e.g., 'wav', 'mp3').

    Returns:
        Path: Path to the saved audio file.

    Raises:
        RuntimeError: If the FFmpeg command fails.
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(f".{audio_format}")
    else:
        output_path = Path(output_path).with_suffix(f".{audio_format}")

    # Build the FFmpeg command to extract audio
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists
        "-i", str(video_path),  # Input video file
        "-vn",  # No video
        "-acodec", "copy" if audio_format == "aac" else "pcm_s16le",  # Use raw PCM for .wav, else default
        str(output_path)
    ]

    try:
        # Execute the FFmpeg command
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")

    return output_path