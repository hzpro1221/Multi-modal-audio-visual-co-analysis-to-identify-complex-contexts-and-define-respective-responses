import torch
from pathlib import Path
from typing import Union, Optional, Literal
import av  # PyAV to check video stream validity

from transformers import AutoProcessor, AutoModelForImageTextToText


class VideoAnswererSmolVLM2:
    """
    Lightweight video understanding using SmolVLM2 (supports video + text → text).
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        dtype: torch.dtype = torch.float16,
        attn_impl: Optional[Literal["eager", "flash_attention_2", "sdpa"]] = "eager",
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            _attn_implementation=attn_impl
        ).to(device)
        self.device = device

    def _is_valid_video(self, video_path: Union[str, Path]) -> bool:
        """
        Check if the video has at least one video stream and can be opened.
        """
        try:
            with av.open(str(video_path)) as container:
                return len(container.streams.video) > 0
        except Exception as e:
            print(f"[⚠️] Invalid video at {video_path}: {e}")
            return False

    def generate(
        self,
        video_path: Union[str, Path],
        prompt: str = "What is happening in this video?",
        max_new_tokens: int = 64
    ) -> str:
        video_path = Path(video_path)
        if not self._is_valid_video(video_path):
            raise ValueError(f"Invalid or unreadable video file: {video_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(video_path)},
                    {"type": "text",  "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=self.model.dtype)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        full_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # ✅ Strip everything before "Assistant:" if present
        if "Assistant:" in full_text:
            return full_text.split("Assistant:")[-1].strip()
        else:
            return full_text