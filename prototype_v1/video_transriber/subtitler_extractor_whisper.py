import torch
import tempfile
from pathlib import Path
from typing import Union

from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class WhisperVideoTranscriber:
    """
    A class that extracts audio from video and transcribes it using Whisper.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        chunk_length_s: int = 60,
        stride_length_s: tuple = (10, 10),
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        """
        Initialize the ASR model, processor, and pipeline.

        Args:
            model_id (str): HuggingFace model ID.
            chunk_length_s (int): Chunk size for audio segmentation.
            stride_length_s (tuple): Overlap in seconds between audio chunks.
            device (str): Device identifier.
            dtype (torch.dtype): Torch data type.
        """
        self.model_id = model_id
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.device = device
        self.dtype = dtype

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager"
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
            return_timestamps="word",
            torch_dtype=self.dtype,
            device=self.device,
        )

    def transcribe_audio(self, audio_path: Union[str, Path]) -> dict:
        """
        Transcribes an audio file using the ASR pipeline.

        Args:
            audio_path (str|Path): Path to the audio file.

        Returns:
            dict: Transcription result.
        """
        return self.pipeline(str(audio_path))

    def transcribe_video(self, video_path: Union[str, Path]) -> str:
        """
        Extracts audio from video, runs ASR, and returns text.

        Args:
            video_path (str|Path): Path to the video file.

        Returns:
            str: Transcribed text.
        """
        with VideoFileClip(str(video_path)) as video, tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_audio:
            video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
            result = self.transcribe_audio(temp_audio.name)
            return result["text"]