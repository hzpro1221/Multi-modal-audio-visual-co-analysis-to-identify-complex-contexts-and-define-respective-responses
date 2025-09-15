import subprocess
from pathlib import Path
from typing import Union, List
from pathlib import Path
from typing import List, Union
import subprocess
import cv2
from PIL import Image

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
    video_path = Path(video_path)
    output_subfolder = Path(output_folder) / video_path.stem
    output_subfolder.mkdir(parents=True, exist_ok=True)

    # Lấy tổng số frame bằng ffprobe
    cmd_info = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-print_format", "default=nokey=1:noprint_wrappers=1",
        str(video_path)
    ]
    try:
        total_frames = int(subprocess.check_output(cmd_info).decode().strip())
    except Exception as e:
        raise RuntimeError(f"❌ Cannot get total frames for video {video_path}: {e}")

    if total_frames == 0:
        raise RuntimeError(f"❌ No frames found in video: {video_path}")

    if num_frame > total_frames:
        num_frame = total_frames

    # Tính vị trí frame đều nhau
    positions = [int((i + 1) * total_frames / (num_frame + 1)) for i in range(num_frame)]
    saved_paths: List[Path] = []

    for i, frame_idx in enumerate(positions):
        out_path = output_subfolder / f"frame_{i+1:03d}.jpg"
        # FFmpeg: trích xuất frame thứ N
        cmd_extract = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{frame_idx})",
            "-vframes", "1",
            "-q:v", "2",  # chất lượng JPEG cao
            str(out_path),
            "-y"  # overwrite nếu tồn tại
        ]
        try:
            subprocess.run(cmd_extract, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            saved_paths.append(out_path)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Cannot extract frame {frame_idx} from {video_path}, skipping. Error: {e}")

    return saved_paths