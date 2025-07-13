import torch
import soundfile as sf
from pathlib import Path
from typing import Union, Optional, Literal

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class VideoAnswererGwenOmni:
    """
    General-purpose video understanding module using Qwen2.5-Omni.

    This class is designed to process video (and optionally audio) inputs and answer user-defined
    prompts or questions based on the content. It leverages multimodal capabilities of the Qwen2.5-Omni
    model to reason about what is seen and/or heard in a video.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-3B",
        dtype: Optional[torch.dtype] = torch.float16,
        device_map: Union[str, dict] = "auto",
        attn_impl: Optional[Literal["eager", "flash_attention_2", "sdpa"]] = "eager",
    ):
        """
        Initializes the Qwen2.5-Omni model and its processor.

        Args:
            model_name (str): HuggingFace model identifier.
            dtype (torch.dtype): Precision to load the model with (e.g., float16 or "auto").
            device_map (str|dict): Device placement strategy.
            attn_impl (str): Attention backend implementation.
        """
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype or "auto",
            device_map=device_map,
            attn_implementation=attn_impl,
            use_safetensors=True,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    def generate(
        self,
        video_path: Union[str, Path],
        prompt: str = "What is happening in this video?",
        include_audio: bool = True,
        max_new_tokens: int = 256,
        return_audio: bool = False,
        audio_output_path: Optional[Union[str, Path]] = "output.wav"
    ) -> str:
        """
        Answers a user-specified question or instruction about a video.

        Args:
            video_path (str|Path): Path to the input video file.
            prompt (str): Natural language instruction or question about the video.
            include_audio (bool): Whether to include audio from the video in reasoning.
            max_new_tokens (int): Maximum number of tokens to generate.
            return_audio (bool): If True, generate and save audio output alongside text.
            audio_output_path (str|Path): Path to save audio file if enabled.

        Returns:
            str: The generated text response based on video and prompt.
        """
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": prompt}
                ],
            },
        ]

        # Format conversation and extract multimodal information
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=include_audio)

        # Tokenize and prepare model input
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=include_audio
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Generate output (text and optionally audio)
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=include_audio, max_new_tokens=max_new_tokens)

        # Decode textual output
        output_text = self.processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Save audio response if requested
        if return_audio and audio_output_path:
            sf.write(
                str(audio_output_path),
                audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )

        return output_text.strip()
