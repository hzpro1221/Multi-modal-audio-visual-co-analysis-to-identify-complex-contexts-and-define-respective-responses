from preprocessor import load_vector_database_from_json
from audio_text_embedder import get_audio_text_similarity
from image_text_embedder import get_image_text_similarity
from video_text_embedder import get_video_text_similarity

import subprocess
import os

if __name__ == "__main__":
    print("\n🔍 Loading vector database...")
    vector_db = load_vector_database_from_json("vector_database.json")
    while True:
        query = input("💬 Enter your query: ")
        best_match_path = ""
        best_score = float("-inf")

        for video_entry in vector_db:
            total_score = 0.0

            # Audio-text similarity
            for model_name, audio_embedding in video_entry["audio_embedding"].items():
                total_score += get_audio_text_similarity(audio_embedding, query, model=model_name)

            # Video-text similarity
            for model_name, video_embedding in video_entry["video_embedding"].items():
                total_score += get_video_text_similarity(video_embedding, query, model=model_name)

            # Keyframe (image)-text similarity
            for model_name, frame_embeddings in video_entry["keyframe_embedding"].items():
                frame_scores = [
                    get_image_text_similarity(frame_embed, query, model=model_name)
                    for frame_embed in frame_embeddings
                ]
                if frame_scores:
                    total_score += sum(frame_scores) / len(frame_scores)

            # Update best match
            if total_score > best_score:
                best_score = total_score
                best_match_path = video_entry["video_path"]

        print(f"🎯 Best matched video: {best_match_path} (Score: {best_score:.4f})")
        
        # Open video with default player
        abs_path = os.path.abspath(best_match_path)
        subprocess.Popen(["xdg-open", abs_path])