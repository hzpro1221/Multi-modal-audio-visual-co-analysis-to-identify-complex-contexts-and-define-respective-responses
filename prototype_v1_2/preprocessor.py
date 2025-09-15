from utils import extract_audio_from_video, extract_keyframe_from_video

from audio_text_embedder import get_audio_embeddings
from video_text_embedder import get_video_embedding
from image_text_embedder import get_image_embedding
from text_text_embedder import get_text_embedding

import json
import shutil
import os

import torch

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
        clusters = json.load(f)

    for cluster in clusters:
        cluster["cluster_name"]["vector"] = try_tensor(cluster["cluster_name"]["vector"])
        
        for video in cluster["videos"]:
            # Convert audio embeddings
            for key, val in video["audio_embedding"].items():
                video["audio_embedding"][key] = try_tensor(val)

            # Convert video embeddings
            for key, val in video["video_embedding"].items():
                video["video_embedding"][key] = try_tensor(val)

            # Convert keyframe embeddings (list of tensors)
            for key, val_list in video["keyframe_embedding"].items():
                video["keyframe_embedding"][key] = [try_tensor(v) for v in val_list]
            
            for chunk in video["chunks"]:
                chunk["vector"] = try_tensor(chunk["vector"])
    return clusters

def save_vector_database_to_json(vector_database, output_path="testset_vector_database.json"):
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
    text_embed_model = ["BGE-3"]

    audio_output_folder = "tmp/audio"
    keyframe_output_folder = "tmp/keyframe"

    # Remove folder and its contents, then recreate empty folder
    shutil.rmtree(audio_output_folder)
    os.makedirs(audio_output_folder)

    shutil.rmtree(keyframe_output_folder)
    os.makedirs(keyframe_output_folder)

    # Load test video
    with open('testset_metadata.json', 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    
    for cluster in clusters:
        # Embedd cluster_name into vector 
        cluster_name = cluster["cluster_name"]
        cluser_name_vector = get_text_embedding(cluster_name)

        del cluster["cluster_name"]
        cluster["cluster_name"] = {
            "name": cluster_name,
            "vector": cluser_name_vector  
        }
        
        for video in cluster["videos"]:
            video["audio_embedding"] = {}
            video["video_embedding"] = {}
            video["keyframe_embedding"] = {}

            audio_path = extract_audio_from_video(
                video_path=video["video_path"],
                output_path=audio_output_folder
            )

            for model in audio_embed_model:
                audio_embed = get_audio_embeddings(audio_path=audio_path, model=model)
                video["audio_embedding"][model] = audio_embed
         
            print("Loading video:", video["video_path"])
            print("Exists:", os.path.exists(video["video_path"]))
            print("Extension:", os.path.splitext(video["video_path"])[1])
            for model in video_embed_model:
                video_embed = get_video_embedding(video_path=video["video_path"], model=model)
                video["video_embedding"][model] = video_embed 

            keyframe_paths = extract_keyframe_from_video(
                video_path=video["video_path"],
                num_frame=16,
                output_folder=keyframe_output_folder
            )

            for model in image_embed_model:
                video["keyframe_embedding"][model] = []
                for keyframe_path in keyframe_paths:
                    img_embed = get_image_embedding(image_path=keyframe_path, model=model)
                    video["keyframe_embedding"][model].append(img_embed)
            
            chunk_embedds = []
            # Embedding text chunk
            for chunk_text in video["chunks"]:
                chunk_embedd = get_text_embedding(chunk_text)
                chunk_embedds.append(
                    {
                        "text": chunk_text,
                        "vector": chunk_embedd 
                    }
                )
            del video["chunks"]
            video["chunks"] = chunk_embedds 

    save_vector_database_to_json(
        vector_database=clusters,
        output_path="vector_database.json"
    )

