import argparse
from subtitler_extractor_whisper import WhisperVideoTranscriber

def video_transribder(video_path: str, model: str = "whisper") -> str:
    """
    This function initializes the VideoTranscriber class to handle video transcription tasks.
    It can be used to transcribe audio from video files into text.
    """    
    if model == "whisper":
        transcriber = WhisperVideoTranscriber()
        return transcriber.transcribe_video(video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe video to text.")
    parser.add_argument("--video_path", required=True, help="Path to the video file.")
    parser.add_argument("--model", default="whisper", help="Model to use (default: whisper).")

    args = parser.parse_args()

    text = video_transribder(video_path=args.video_path, model=args.model)
    print("📝 Transcription:\n", text)