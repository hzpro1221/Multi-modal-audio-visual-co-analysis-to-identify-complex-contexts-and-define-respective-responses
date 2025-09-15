from preprocessor import load_vector_database_from_json
from audio_text_embedder import get_audio_text_similarity
from image_text_embedder import get_image_text_similarity
from video_text_embedder import get_video_text_similarity
from text_text_embedder import get_text_text_similarity

import os
import random
import subprocess
import time

print("\n🔍 Loading vector database...")
clusters = load_vector_database_from_json("vector_database.json")
cluster_embeddings = [cluster["cluster_name"] for cluster in clusters]

def content_retrieve(query):
    # Step 1: Predict cluster
    max_score = -float("inf")
    predicted_cluster = None

    for cluster in clusters:
        score = get_text_text_similarity(
            topic_embed=cluster["cluster_name"]["vector"],
            query=query
        )
        if score > max_score:
            max_score = score
            predicted_cluster = cluster

    # Step 2: Predict video in cluster
    video_scores = []                
    for video in predicted_cluster["videos"]:
        total_score = 0
        for model_name, audio_embedding in video.get("audio_embedding", {}).items():
            total_score += get_audio_text_similarity(audio_embedding, query, model=model_name)

        for model_name, video_embedding in video.get("video_embedding", {}).items():
            total_score += get_video_text_similarity(video_embedding, query, model=model_name)

        for model_name, frame_embeddings in video.get("keyframe_embedding", {}).items():
            frame_scores = [
                get_image_text_similarity(frame_embed, query, model=model_name)
                for frame_embed in frame_embeddings
            ]
            if frame_scores:
                total_score += sum(frame_scores) / len(frame_scores)

        video_scores.append((video["video_path"], total_score))        
    
    queried_video_path = video_scores[0][0]
    queried_video_score = video_scores[0][1]

    return queried_video_path, queried_video_score 

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    video_path, video_score = content_retrieve("Today, I will talk about Machine learning")
    print("out_video: ", video_path, "; video_score: ", video_score) 

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")    