import torch
from pathlib import Path
from typing import Union, Optional, Literal

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

    def generate(
        self,
        video_path: Union[str, Path],
        prompt: str = "What is happening in this video?",
        max_new_tokens: int = 64
    ) -> str:
        """
        Answers a question or instruction based on video content.

        Args:
            video_path (str|Path): Path to the input video file (.mp4, etc.)
            prompt (str): Question or instruction about the video.
            max_new_tokens (int): Max number of tokens to generate.

        Returns:
            str: Generated textual response.
        """
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
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
