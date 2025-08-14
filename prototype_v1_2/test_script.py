from preprocessor import load_vector_database_from_json
from audio_text_embedder import get_audio_text_similarity
from image_text_embedder import get_image_text_similarity
from video_text_embedder import get_video_text_similarity
from text_text_embedder import get_text_text_similarity

import os
import random
import subprocess

if __name__ == "__main__":
    print("\n🔍 Loading vector database...")
    clusters = load_vector_database_from_json("vector_database.json")

    total = 0
    correct = 0  

    cluster_embeddings = [cluster["cluster_name"] for cluster in clusters]

    for cluster_idx, cluster in enumerate(clusters):
        cluster_name = cluster["cluster_name"]["name"]
        print(f"\n📂 Cluster {cluster_idx + 1}/{len(clusters)}: '{cluster_name}'")

        for video_idx, video in enumerate(cluster["videos"]):
            video_name = video["video_name"]
            print(f"  🎞 Video {video_idx + 1}/{len(cluster['videos'])}: '{video_name}'")

            for chunk_idx, chunk in enumerate(video["chunks"]):
                query = chunk["text"]
                print(f"    🧩 Chunk {chunk_idx + 1}/{len(video['chunks'])}: \"{query[:60]}...\"")

                # Step 1: Predict cluster
                max_score = -float("inf")
                predicted_cluster = ""

                for embed in cluster_embeddings:
                    score = get_text_text_similarity(
                        topic_embed=embed["vector"],
                        query=query
                    )
                    if score > max_score:
                        max_score = score
                        predicted_cluster = embed["name"]

                if predicted_cluster != cluster_name:
                    print(f"      ❌ Wrong cluster: predicted '{predicted_cluster}', expected '{cluster_name}'")
                    total += 1
                    continue
                else:
                    print(f"      ✅ Correct cluster match")

                # Step 2: Predict video in cluster
                video_scores = []

                for to_query_video in cluster["videos"]:
                    total_score = 0

                    for model_name, audio_embedding in to_query_video.get("audio_embedding", {}).items():
                        total_score += get_audio_text_similarity(audio_embedding, query, model=model_name)

                    for model_name, video_embedding in to_query_video.get("video_embedding", {}).items():
                        total_score += get_video_text_similarity(video_embedding, query, model=model_name)

                    for model_name, frame_embeddings in to_query_video.get("keyframe_embedding", {}).items():
                        frame_scores = [
                            get_image_text_similarity(frame_embed, query, model=model_name)
                            for frame_embed in frame_embeddings
                        ]
                        if frame_scores:
                            total_score += sum(frame_scores) / len(frame_scores)

                    video_scores.append((to_query_video["video_name"], total_score))

                # Sort videos by score descending
                video_scores.sort(key=lambda x: x[1], reverse=True)

                # Check if ground-truth video is in top 2
                top_2_videos = [name for name, _ in video_scores[:2]]

                if video_name in top_2_videos:
                    print(f"      ✅ Correct video match (Top-2)")
                    correct += 1
                else:
                    print(f"      ❌ Wrong video: top 2 predicted {[v for v in top_2_videos]}, expected '{video_name}'")

                total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n🎯 Final Accuracy: {correct}/{total} correct ({accuracy:.2f}%)")
