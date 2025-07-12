
from typing import Union, Dict, List, Optional
from pathlib import Path
import subprocess
import tempfile

from utils import fixed_interval_video_splitter

def semantic_video_info_extractor(
    video_path: Union[str, Path],
    questions: Optional[List[str]] = None,
    transcribe: bool = True,
    summary: bool = True,
    answer_questions: bool = True,
    max_new_tokens: int = 64,
    transcribe_model: str = "whisper",
    video_understand_model: str = "smolvlm2"
) -> Dict[str, Optional[Union[str, List[Dict], Dict[str, str]]]]:
    """
    Extracts semantic information from a video using multimodal analysis.

    Args:
        video_path (str|Path): Path to the input video.
        questions (List[str], optional): User-defined questions to answer based on the video.
        transcribe (bool): If True, transcribes the video's audio into text.
        summary (bool): If True, generates a general summary of the video content.
        answer_questions (bool): If True, answers the provided questions using video understanding.
        max_new_tokens (int): Maximum number of tokens for the model to generate in responses.
        transcribe_model (str): Model to use for transcription (default: "whisper").
        video_understand_model (str): Model to use for video understanding (default: "smolvlm2").

    Returns:
        Dict: A dictionary with keys that may include:
              - "summary": General description of the video.
              - "subtitles": Transcript of the audio content.
              - "questions": A dict mapping each question to its answer.
    """
    results: Dict[str, Optional[Union[str, List[Dict], Dict[str, str]]]] = {}
    video_path = str(video_path)

    results["video_path"] = video_path
    if transcribe:
        results["subtitles"] = subprocess.run([
            "python", "prototype_v1/video_transriber/main.py",
            "--video_path", video_path,
            "--model", transcribe_model
            ], capture_output=True, text=True)

    results = {
        "summary": [],
        "questions": {q: [] for q in questions} if answer_questions and questions else {}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        video_chunks = fixed_interval_video_splitter(
            video_path=video_path,
            output_dir=tmpdir,
            interval_sec=5,  # Split every 5 seconds
            output_format="mp4",
            prefix="chunk"
        )
        for chunk_path in video_chunks:
            prompts = []
            if summary:
                summary_prompt = (
                    "Provide a concise and informative summary of the video, including main events, topics, "
                    "key visual or audio elements, and its overall purpose or message."
                )
                prompts.append(summary_prompt)

            if answer_questions and questions:
                for q in questions:
                    prompts.append(q)

            result = subprocess.run([
                "python", "prototype_v1/video_understander/main.py",
                "--video_path", chunk_path,
                "--prompt", prompts,
                "--max_new_tokens", str(max_new_tokens),
                "--model", video_understand_model
            ], capture_output=True, text=True)

            if summary:
                results["summary"].append(result[summary_prompt].stdout.strip())            

            if answer_questions and questions:
                results["questions"][q].append(result[q].stdout.strip())                

    return results