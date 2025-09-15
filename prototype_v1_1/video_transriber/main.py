import argparse
import json
from subtitler_extractor_whisper import WhisperVideoTranscriber

def video_transcriber(video_path: str, model: str = "whisper") -> str:
    """
    This function initializes the VideoTranscriber class to handle video transcription tasks.
    It can be used to transcribe audio from video files into text.
    """    
    if model == "whisper":
        transcriber = WhisperVideoTranscriber()
        return transcriber.transcribe_video(video_path)
    else:
        raise ValueError(f"Unsupported model: {model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe video to text.")
    parser.add_argument("--video_path", required=True, help="Path to the video file.")
    parser.add_argument("--model", default="whisper", help="Model to use (default: whisper).")
    parser.add_argument("--output_file", required=True, help="Path to the output JSON file.")

    args = parser.parse_args()

    text = video_transcriber(video_path=args.video_path, model=args.model)

    # ✅ Write transcript to JSON file
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"transcript": text}, f, ensure_ascii=False, indent=2)
