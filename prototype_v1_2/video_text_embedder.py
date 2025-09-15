from embedders.XCLIP import XCLIPWrapper
from embedders.Clip4Clip import Clip4ClipWrapper 

import numpy as np
from torch import Tensor
# Initialize models
xclip = XCLIPWrapper()
clip4clip = Clip4ClipWrapper()

def get_video_embedding(video_path: str, model: str = "XCLIP") -> Tensor:
    """
    Generate a video embedding using the specified model.

    :param video_path: Path to the video file (e.g., ".mp4" or ".avi").
    :param model: Name of the embedding model to use. Options are 'XCLIP' or 'Clip4Clip'.
    :return: a Tensor representing the video embedding.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "XCLIP":
        # print("🎥 Using X-CLIP model for video embedding.")
        return xclip.get_video_embedding(video_path)
    elif model == "CLIP4CLIP":
        # print("🎥 Using CLIP4Clip model for video embedding.")
        return clip4clip.get_video_embedding(video_path)
    
    raise ValueError(f"❌ Unsupported model: {model}")

def get_video_text_similarity(video_embedding: Tensor, text: str, model: str = "XCLIP") -> float:
    """
    Compute the similarity score between a video embedding and a text string.

    :param video_embedding: Tensor representing the video embedding.
    :param text: Text string to compare with the video.
    :param model: Name of the embedding model to use. Options are 'XCLIP' or 'Clip4Clip'.
    :return: Cosine similarity score as a float.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "XCLIP":
        # print("🔗 Computing similarity using X-CLIP model...")
        text_embedding = xclip.get_text_embedding(text)
        return xclip.compute_similarity(video_embedding, text_embedding)
    elif model == "CLIP4CLIP":
        # print("🔗 Computing similarity using CLIP4Clip model...")
        text_embedding = clip4clip.get_text_embedding(text)
        return clip4clip.compute_similarity(video_embedding, text_embedding)
    
    raise ValueError(f"❌ Unsupported model: {model}")

if __name__ == "__main__":
    video_path = "sample_video.mp4"  # Replace with a valid test video path
    text_queries = [
        "a dog playing in the park",
        "a cat chasing a laser pointer",
        "a person riding a bicycle"
    ]
    models = ["XCLIP", "CLIP4CLIP"]
    # print("🧪 Starting video-text similarity tests...")
    for model_name in models:
        # print(f"🔍 Testing model: {model_name}")
        try:
            # Get video embedding
            video_embedding = get_video_embedding(video_path, model=model_name)
            assert isinstance(video_embedding, Tensor), "Video embedding must be a tensor"
            # print(f"✅ Video embedding shape for {model_name}:", video_embedding.shape)

            # Compute similarity for each text
            for text in text_queries:
                score = get_video_text_similarity(video_embedding, text, model=model_name)
                # print(f"🎯 Similarity score for '{text}': {score:.4f}")

        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
