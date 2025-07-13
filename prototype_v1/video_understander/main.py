import argparse
import json
import os
import logging
import av  # PyAV logging suppression
import sys

from video_answerer_smolvlm2 import VideoAnswererSmolVLM2
from typing import List, Dict


def video_answerer(
        video_path: str,
        prompts: List[str],
        max_new_tokens: int = 64,
        model: str = "smolvlm2"
) -> Dict[str, str]:
    if model == "smolvlm2":
        answerer = VideoAnswererSmolVLM2()
    else:
        raise ValueError(f"Model '{model}' is not supported.")
    
    result = {}
    for prompt in prompts:
        print("prompt:", prompt)
        result[prompt] = answerer.generate(video_path, prompt, max_new_tokens)
        print("result[prompt]:", result[prompt])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer questions about video content.")
    parser.add_argument("--video_path", required=True, help="Path to the input video file.")
    parser.add_argument("--prompt_file", type=str, required=False, help="Path to a JSON file containing a list of prompts/questions.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate.")
    parser.add_argument("--model", default="smolvlm2", help="Model type (default: smolvlm2).")
    parser.add_argument("--output_file", required=True, help="Path to save JSON result.")

    args = parser.parse_args()

    # Read prompts
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise ValueError("The prompt file must contain a JSON list of strings.")
    else:
        prompts = ["What is happening in this video?"]

    response = video_answerer(
        video_path=args.video_path,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        model=args.model
    )

    # ✅ Write JSON response to file
    with open(args.output_file, "w", encoding="utf-8") as out_file:
        json.dump(response, out_file, ensure_ascii=False, indent=2)
