from preprocessor import load_video_embeds
from extent_network import NeuralNetworkRetriever

import os

import torch

model = NeuralNetworkRetriever().to("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/content/model.pth"))

DEVICE = model.device 
VIDEO_EMBEDS_PATH = "video_embeds.json"
GET_VIDEO_EMBEDDER = model._video_combine_embeds
GET_TEXT_EMBEDDER =  model._text_combine_embeds


print("\nLoading video embeds...")
video_embeds = load_video_embeds(VIDEO_EMBEDS_PATH, device=DEVICE)

def content_retrieve(query):


    print("Getting video embed...\n")
    candidate_vids = torch.cat([video_embed.unsqueeze(0) for video_path, video_embed in video_embeds.items()])
    print(f"candidate_vids.device: {candidate_vids.device}")
    print(f"model.device: {model.device}")
    print(f"query: {query}")
    video_embeds_output = model.get_video_embed(candidate_vids.unsqueeze(0))
    
    print("Getting text embed...\n")
    text_embed_output = model.get_text_embed(GET_TEXT_EMBEDDER([query]))

    sim = model.compute_similarity(text_embed_output, video_embeds_output)
    topk_values, topk_indices = sim.topk(3, dim=-1)  # along candidate dim

    keys = list(video_embeds.keys())
    for i, (score, idx) in enumerate(zip(topk_values[0], topk_indices[0])):
        video_path = keys[idx]
        filename = os.path.basename(video_path)
        print(f"Video: {filename}, Similarity Score: {score.item():.4f}")
    
    best_video_path = keys[topk_indices[0][0]]
    best_score = topk_values[0][0].item()
    
    return best_video_path, best_score

if __name__ == "__main__":
  print(content_retrieve("what is the meaning of life?"))

# clusters = load_vector_database_from_json("video_embeds.json")
# cluster_embeddings = [cluster["cluster_name"] for cluster in clusters]

# def content_retrieve(query):
#     # Step 1: Predict cluster
#     max_score = -float("inf")
#     predicted_cluster = None

#     for cluster in clusters:
#         score = get_text_text_similarity(
#             topic_embed=cluster["cluster_name"]["vector"],
#             query=query
#         )
#         if score > max_score:
#             max_score = score
#             predicted_cluster = cluster

#     # Step 2: Predict video in cluster
#     video_scores = []                
#     for video in predicted_cluster["videos"]:
#         total_score = 0
#         for model_name, audio_embedding in video.get("audio_embedding", {}).items():
#             total_score += get_audio_text_similarity(audio_embedding, query, model=model_name)

#         for model_name, video_embedding in video.get("video_embedding", {}).items():
#             total_score += get_video_text_similarity(video_embedding, query, model=model_name)

#         for model_name, frame_embeddings in video.get("keyframe_embedding", {}).items():
#             frame_scores = [
#                 get_image_text_similarity(frame_embed, query, model=model_name)
#                 for frame_embed in frame_embeddings
#             ]
#             if frame_scores:
#                 total_score += sum(frame_scores) / len(frame_scores)

#         video_scores.append((video["video_path"], total_score))        
    
#     queried_video_path = video_scores[0][0]
#     queried_video_score = video_scores[0][1]

#     return queried_video_path, queried_video_score 

# if __name__ == "__main__":
#     start_time = time.perf_counter()
    
#     video_path, video_score = content_retrieve("Today, I will talk about Machine learning")
#     print("out_video: ", video_path, "; video_score: ", video_score) 

#     end_time = time.perf_counter()
#     print(f"Execution time: {end_time - start_time:.4f} seconds")    