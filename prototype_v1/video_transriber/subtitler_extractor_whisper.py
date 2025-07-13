import torch
import tempfile
from pathlib import Path
from typing import Union
import traceback

from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class WhisperVideoTranscriber:
    """
    A class that extracts audio from video and transcribes it using Whisper.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        chunk_length_s: int = 20,
        stride_length_s: tuple = (5, 5),
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        print("🧠 Initializing WhisperVideoTranscriber...")
        self.model_id = model_id
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.device = device
        self.dtype = dtype

        try:
            print(f"📥 Loading model: {model_id}")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="eager"
            ).to(self.device)
            print("✅ Model loaded.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            raise

        try:
            print("📦 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            print("✅ Processor loaded.")
        except Exception as e:
            print(f"❌ Failed to load processor: {e}")
            traceback.print_exc()
            raise

        try:
            print("⚙️ Building transcription pipeline...")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=self.chunk_length_s,
                stride_length_s=self.stride_length_s,
                torch_dtype=self.dtype,
                device=self.device,
            )
            print("🚀 Pipeline ready.\n")
        except Exception as e:
            print(f"❌ Failed to create pipeline: {e}")
            traceback.print_exc()
            raise

    def transcribe_audio(self, audio_path: Union[str, Path]) -> dict:
        print(f"🎧 Transcribing audio: {audio_path}")
        try:
            result = self.pipeline(str(audio_path))
            print("📝 Transcription complete.\n")
            return result
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            traceback.print_exc()
            raise

    def transcribe_video(self, video_path: Union[str, Path]) -> str:
        print(f"🎬 Extracting audio from video: {video_path}")
        try:
            with VideoFileClip(str(video_path)) as video, tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_audio:
                video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                print(f"🎚️ Audio extracted to temp file: {temp_audio.name}")
                result = self.transcribe_audio(temp_audio.name)
                print("📄 Final transcription result:")
                print(result["text"])
                return result["text"]
        except Exception as e:
            print(f"❌ Failed to extract or transcribe video: {e}")
            traceback.print_exc()
            raise
