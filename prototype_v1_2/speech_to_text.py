import torch
import os
import json
from pathlib import Path

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_ID = "openai/whisper-large-v3"
CHUNK_LENGTH_S = 30
STRIDE_LENGTH_S = (5, 5)
SPLITTERS = {".", "!", "?"}
TOKEN_LIMIT = 40

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="eager",
).to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=CHUNK_LENGTH_S,
    stride_length_s=STRIDE_LENGTH_S,
    return_timestamps="word",
    torch_dtype=DTYPE,
    device=DEVICE,
)

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def is_sensitive_word_only(word: str) -> bool:
    return False  # Placeholder logic

def remove_call_to_action_sentences(text: str) -> str:
    return text  # Placeholder logic

def transcribe_audio(path: str) -> dict:
    """Run ASR pipeline on audio file path."""
    return asr_pipeline(path)

def convert_to_json_segments(data: list, token_limit: int = TOKEN_LIMIT) -> list:
    """
    Convert ASR output into a list of subtitle segments in dictionary format.
    Each segment has: 'timestamp': [start, end], 'text': content.
    """
    segments = []
    current = []
    start = end = prev_end = None

    for token in data:
        text = token["text"].strip()
        ts = token["timestamp"]

        if ts[0] is None or ts[1] is None:
            continue

        # Time-based split
        if prev_end is not None and ts[0] - prev_end > 0.5 and current:
            sentence = " ".join(current)
            if not (len(current) == 1 and is_sensitive_word_only(current[0])):
                segments.append({
                    "timestamp": [format_timestamp(start), format_timestamp(prev_end)],
                    "text": remove_call_to_action_sentences(sentence)
                })
            current, start, end = [], None, None

        if start is None:
            start = ts[0]
        end = ts[1]
        prev_end = end
        current.append(text)

        # Content-based split
        if text[-1] in SPLITTERS or len(current) >= token_limit:
            sentence = " ".join(current)
            if not (len(current) == 1 and is_sensitive_word_only(current[0])):
                segments.append({
                    "timestamp": [format_timestamp(start), format_timestamp(end)],
                    "text": remove_call_to_action_sentences(sentence)
                })
            current, start, end, prev_end = [], None, None, None

    # Final group
    if current:
        sentence = " ".join(current)
        segments.append({
            "timestamp": [format_timestamp(start), format_timestamp(end)],
            "text": remove_call_to_action_sentences(sentence)
        })

    return segments

def speech_to_text(audio_path: str) -> list:
    """Transcribe audio and return structured subtitle segments."""
    print("🔍 Transcribing audio from file...")
    transcription_result = transcribe_audio(audio_path)

    print("📝 Transcription complete. Parsing segments...")
    print(transcription_result.get("text", ""))
    
    segments = convert_to_json_segments(transcription_result["chunks"])
    return segments


if __name__ == "__main__":
    input_audio = "test_output.wav"

    if not os.path.exists(input_audio):
        print(f"❌ Audio file not found: {input_audio}")
        exit(1)

    print("🚀 Running speech-to-text pipeline...")
    segments = speech_to_text(input_audio)

    print("\n--- Preview ---")
    for seg in segments:
        print(seg)
