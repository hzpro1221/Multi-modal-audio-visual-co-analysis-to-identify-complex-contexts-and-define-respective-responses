from utils import extract_audio_from_video, extract_keyframe_from_video

from extent_network import NeuralNetworkRetriever
import json
import shutil
import os
from pathlib import Path

import torch

def save_video_embeds(video_embeds: dict, json_path: str):
    serializable = {str(k): v.cpu().tolist() for k, v in video_embeds.items()}
    with open(json_path, "w") as f:
        json.dump(serializable, f)

def load_video_embeds(json_path: str, device: str = "cpu") -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    return {k: torch.tensor(v, device=device) for k, v in data.items()}

if __name__ == "__main__":
    print("load model...")
    model = NeuralNetworkRetriever().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("/content/model.pth"))

    EVALUATE_FOLDER = "/content/Multi-modal-audio-visual-co-analysis-to-identify-complex-contexts-and-define-respective-responses/prototype_v1_2/evaluate_videos"
    GET_VIDEO_EMBEDDER = model._video_combine_embeds
    GET_TEXT_EMBEDDER =  model._text_combine_embeds

    video_paths = list(Path(EVALUATE_FOLDER).rglob("*.mp4"))
    print(video_paths)
    video_embeds = {}
    print("preprocess video...")

    for video_path in video_paths:
        if video_path not in video_embeds:
            emb = GET_VIDEO_EMBEDDER([video_path]).squeeze(0)
            video_embeds[video_path] = emb
    print(video_embeds)

    print("save video embeds...")
    save_video_embeds(video_embeds, "video_embeds.json")