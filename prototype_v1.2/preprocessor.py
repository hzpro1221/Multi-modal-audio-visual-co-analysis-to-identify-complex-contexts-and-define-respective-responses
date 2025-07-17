from utils import extract_audio_from_video, extract_keyframe_from_video

from audio_text_embedder import get_audio_embeddings
from video_text_embedder import get_video_embedding
from image_text_embedder import get_image_embedding

import json

import torch
import numpy as np
import json

def load_vector_database_from_json(input_path="vector_database.json"):
    def try_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        else:
            try:
                return torch.tensor(obj, dtype=torch.float32)
            except Exception:
                pass
        return obj  # Leave as-is if not convertible

    with open(input_path, "r") as f:
        data = json.load(f)

    for item in data:
        # Convert audio embeddings
        if "audio_embedding" in item:
            for key, val in item["audio_embedding"].items():
                item["audio_embedding"][key] = try_tensor(val)

        # Convert video embeddings
        if "video_embedding" in item:
            for key, val in item["video_embedding"].items():
                item["video_embedding"][key] = try_tensor(val)

        # Convert keyframe embeddings (list of tensors)
        if "keyframe_embedding" in item:
            for key, val_list in item["keyframe_embedding"].items():
                item["keyframe_embedding"][key] = [try_tensor(v) for v in val_list]

    return data

def save_vector_database_to_json(vector_database, output_path="vector_database.json"):
    def serialize_tensor(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(vector_database, f, indent=4, default=serialize_tensor)

if __name__ == "__main__":
    audio_embed_model = ["CLAP"]
    video_embed_model = ["XCLIP", "Clip4Clip"]
    image_embed_model = ["CLIP", "ALIGN"]

    audio_output_folder = "video/audio"
    keyframe_output_folder = "video/keyframe"

    # Video path to add to video_database
    video_paths = [
        "sample_video.mp4"
    ]

    vector_database = []
    for video_path in video_paths:
        video_vector = {
            "video_path": video_path,
            "audio_embedding": {},
            "video_embedding": {},
            "keyframe_embedding": {}
        }

        audio_path = extract_audio_from_video(
            video_path=video_path,
            output_path=audio_output_folder
        )

        for model in audio_embed_model:
            audio_embed = get_audio_embeddings(audio_path=audio_path, model=model)
            video_vector["audio_embedding"][model] = audio_embed
         
        for model in video_embed_model:
            video_embed = get_video_embedding(video_path=video_path, model=model)
            video_vector["video_embedding"][model] = video_embed 

        keyframe_paths = extract_keyframe_from_video(
            video_path=video_path,
            num_frame=16,
            output_folder=keyframe_output_folder
        )

        for model in image_embed_model:
            video_vector["keyframe_embedding"][model] = []
            for keyframe_path in keyframe_paths:
                img_embed = get_image_embedding(image_path=keyframe_path, model=model)
                video_vector["keyframe_embedding"][model].append(img_embed)
        
        vector_database.append(video_vector)
    
    save_vector_database_to_json(
        vector_database=vector_database,
        output_path="vector_database.json"
    )

