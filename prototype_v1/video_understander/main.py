import argparse
from video_answerer_smolvlm2 import VideoAnswererSmolVLM2
from typing import List

def video_answerer(
        video_path: str,
        prompts: List[str] = ["What is happening in this video?"],
        max_new_tokens: int = 64,
        model: str = "smolvlm2"
) -> str:
    """
    This function initializes the VideoAnswererSmolVLM2 class to handle video understanding tasks.
    It can be used to answer questions about video content.
    """
    result = {}
    for prompt in prompts:
        if model == "smolvlm2":
            answerer = VideoAnswererSmolVLM2()
            result[prompt] = answerer.generate(video_path, prompt, max_new_tokens)
        else:
            raise ValueError(f"Model '{model}' is not supported.")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer questions about video content.")
    parser.add_argument("--video_path", required=True, help="Path to the input video file.")
    parser.add_argument("--prompts", default=["What is happening in this video?"], help="Prompts/questions to ask.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate.")
    parser.add_argument("--model", default="smolvlm2", help="Model type (default: smolvlm2).")

    args = parser.parse_args()

    response = video_answerer(
        video_path=args.video_path,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        model=args.model
    )
    print("🗣️ Answer:", response)
