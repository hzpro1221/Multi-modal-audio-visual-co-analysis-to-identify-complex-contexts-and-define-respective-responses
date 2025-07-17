import subprocess
from pathlib import Path
from typing import Union, List

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

def extract_keyframe_from_video(
    video_path: Union[str, Path],
    num_frame: int = 16,
    output_folder: Union[str, Path] = "keyframes"
) -> List[Path]:
    """
    Extracts keyframes from a video using FFmpeg.

    Args:
        video_path (str | Path): Path to the input video file.
        num_frame (int): Number of keyframes to extract evenly across the video.
        output_folder (str | Path): Directory where keyframes will be saved.

    Returns:
        List[Path]: List of paths to the extracted keyframe image files.

    Raises:
        RuntimeError: If the FFmpeg command fails.
    """
    video_path = Path(video_path)
    output_subfolder = Path(output_folder) / video_path.stem
    output_subfolder.mkdir(parents=True, exist_ok=True)

    output_pattern = output_subfolder / "frame_%03d.jpg"

    # FFmpeg filter to extract evenly spaced frames
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"select='not(mod(n,ceil(n/{num_frame})))'",
        "-vsync", "vfr",
        str(output_pattern)
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")

    return sorted(output_subfolder.glob("frame_*.jpg"))
