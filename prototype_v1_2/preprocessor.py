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

    EVALUATE_FOLDER = "evaluate_videos"
    GET_VIDEO_EMBEDDER = model._video_combine_embeds
    GET_TEXT_EMBEDDER =  model._text_combine_embeds

    video_paths = [p.relative_to(EVALUATE_FOLDER) for p in Path(EVALUATE_FOLDER).rglob("*.mp4")]
    video_embeds = {}
    print("preprocess video...")

    for video_path in video_paths:
        if video_path not in video_embeds:
            emb = GET_VIDEO_EMBEDDER([video_path]).squeeze(0)
            video_embeds[video_path] = emb
    print(video_embeds)

    print("save video embeds...")
    save_video_embeds(video_embeds, "video_embeds.json")

    # audio_output_folder = "tmp/audio"
    # keyframe_output_folder = "tmp/keyframe"

    # # Remove folder and its contents, then recreate empty folder
    # shutil.rmtree(audio_output_folder)
    # os.makedirs(audio_output_folder)

    # shutil.rmtree(keyframe_output_folder)
    # os.makedirs(keyframe_output_folder)

    # # Load test video
    # with open('testset_metadata.json', 'r', encoding='utf-8') as f:
    #     clusters = json.load(f)
    
    # for cluster in clusters:
    #     # Embedd cluster_name into vector 
    #     cluster_name = cluster["cluster_name"]
    #     cluser_name_vector = get_text_embedding(cluster_name)

    #     del cluster["cluster_name"]
    #     cluster["cluster_name"] = {
    #         "name": cluster_name,
    #         "vector": cluser_name_vector  
    #     }
        
    #     for video in cluster["videos"]:
    #         video["audio_embedding"] = {}
    #         video["video_embedding"] = {}
    #         video["keyframe_embedding"] = {}

    #         audio_path = extract_audio_from_video(
    #             video_path=video["video_path"],
    #             output_path=audio_output_folder
    #         )

    #         for model in audio_embed_model:
    #             audio_embed = get_audio_embeddings(audio_path=audio_path, model=model)
    #             video["audio_embedding"][model] = audio_embed
         
    #         print("Loading video:", video["video_path"])
    #         print("Exists:", os.path.exists(video["video_path"]))
    #         print("Extension:", os.path.splitext(video["video_path"])[1])
    #         for model in video_embed_model:
    #             video_embed = get_video_embedding(video_path=video["video_path"], model=model)
    #             video["video_embedding"][model] = video_embed 

    #         keyframe_paths = extract_keyframe_from_video(
    #             video_path=video["video_path"],
    #             num_frame=16,
    #             output_folder=keyframe_output_folder
    #         )

    #         for model in image_embed_model:
    #             video["keyframe_embedding"][model] = []
    #             for keyframe_path in keyframe_paths:
    #                 img_embed = get_image_embedding(image_path=keyframe_path, model=model)
    #                 video["keyframe_embedding"][model].append(img_embed)
            
    #         chunk_embedds = []
    #         # Embedding text chunk
    #         for chunk_text in video["chunks"]:
    #             chunk_embedd = get_text_embedding(chunk_text)
    #             chunk_embedds.append(
    #                 {
    #                     "text": chunk_text,
    #                     "vector": chunk_embedd 
    #                 }
    #             )
    #         del video["chunks"]
    #         video["chunks"] = chunk_embedds 

    save_vector_database_to_json(
        vector_database=clusters,
        output_path="vector_database.json"
    )

