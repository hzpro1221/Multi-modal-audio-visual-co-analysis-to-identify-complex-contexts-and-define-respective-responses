from typing import Union, Dict, List, Optional
from pathlib import Path
import subprocess
import json
import cv2

from utils import fixed_interval_video_splitter

def semantic_video_info_extractor(
    video_path: Union[str, Path],
    segments_folder: Union[str, Path],
    questions: Optional[List[str]] = None,
    transcribe: bool = True,
    summary: bool = True,
    answer_questions: bool = True,
    max_new_tokens: int = 64,
    transcribe_model: str = "whisper",
    video_understand_model: str = "smolvlm2",
    segments_len: int = 5
) -> Dict[str, Optional[Union[str, List[Dict], Dict[str, str]]]]:
    results: Dict[str, Optional[Union[str, List[Dict], Dict[str, str]]]] = {}
    video_path = str(video_path)
    segments_folder = Path(segments_folder)
    segments_folder.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print(f"📥 Analyzing video: {video_path}")
    print("========================================")

    results["video_path"] = video_path

    if transcribe:
        print("🔊 Transcribing video audio...")
        transcript_output_file = segments_folder / "transcription_output.json"
        transcript = subprocess.run([
            "python", "prototype_v1/video_transriber/main.py",
            "--video_path", video_path,
            "--model", transcribe_model,
            "--output_file", str(transcript_output_file)
        ], capture_output=True, text=True)

        if transcript.returncode == 0:
            try:
                with open(transcript_output_file, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                results["subtitles"] = parsed.get("transcript", "").strip()
                print("✅ Transcription completed.\n")
                print("📝 Subtitles:\n", results["subtitles"])
            except Exception as e:
                print("⚠️ Failed to parse transcription JSON:", e)
                results["subtitles"] = None
        else:
            print("❌ Transcription failed:")
            print(transcript.stderr)
            results["subtitles"] = None

    results["segments"] = []

    print("🧩 Splitting video into 5s chunks for understanding...")
    video_chunks = fixed_interval_video_splitter(
        video_path=video_path,
        output_dir=segments_folder,
        interval_sec=segments_len,
        output_format="mp4",
        prefix="chunk"
    )
    print(f"📦 Total chunks created: {len(video_chunks)}\n")

    # Get video duration once before loop
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    results["segments"] = []

    for i, chunk_path in enumerate(video_chunks):
        print(f"--- Analyzing chunk {i + 1}/{len(video_chunks)} ---")
        print(f"🗂️  Path: {chunk_path}")

        start_time = round(i * segments_len, 3)
        end_time = round(min((i + 1) * segments_len, duration), 3)

        segment_output = {
            "start": start_time,
            "end": end_time,
            "questions": {}
        }

        # Build prompts
        prompts = []
        if summary:
            summary_prompt = (
                "Provide a concise and informative summary of the video, including main events, topics, "
                "key visual or audio elements, and its overall purpose or message."
            )
            prompts.append(summary_prompt)
        if answer_questions and questions:
            prompts.extend(questions)

        # Save prompt file
        prompt_file = segments_folder / f"prompt_{i}.json"
        with open(prompt_file, "w", encoding="utf-8") as pf:
            json.dump(prompts, pf, ensure_ascii=False, indent=2)

        # Run the understanding subprocess
        output_file = segments_folder / f"output_{i}.json"
        result = subprocess.run([
            "python", "prototype_v1/video_understander/main.py",
            "--video_path", str(chunk_path),
            "--prompt_file", str(prompt_file),
            "--max_new_tokens", str(max_new_tokens),
            "--model", video_understand_model,
            "--output_file", str(output_file)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Failed to understand chunk:")
            print(result.stderr)
            continue

        # Parse results
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            print("📝 Parsed output:", parsed)
        except Exception as e:
            print(f"⚠️ Error parsing output file: {e}")
            continue

        # Assign summary
        if summary and summary_prompt in parsed:
            segment_output["summary"] = parsed[summary_prompt].strip()

        # Assign answers to questions
        if answer_questions and questions:
            for q in questions:
                segment_output["questions"][q] = parsed.get(q, "").strip()

        results["segments"].append(segment_output)
        print("✅ Chunk processed.\n")

    print("🏁 Video analysis complete.")
    return results
