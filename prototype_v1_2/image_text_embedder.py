from embedders.CLIP import CLIPWrapper
from embedders.ALIGN import ALIGNWrapper
import torch

# Initialize models 
clip = CLIPWrapper()
align = ALIGNWrapper()

def get_image_embedding(image_path: str, model: str = "CLIP") -> torch.Tensor:
    """
    Generate an image embedding using the specified model.

    :param image_path: Path to the image file (e.g., ".jpg" or ".png").
    :param model: Name of the embedding model to use. Options are 'CLIP' or 'ALIGN'.
    :return: Torch tensor representing the image embedding.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "CLIP":
        # print("🖼️ Using CLIP model for image embedding.")
        return clip.get_image_embedding(image_path)
    elif model == "ALIGN":
        # print("🖼️ Using ALIGN model for image embedding.")
        return align.get_image_embedding(image_path)
    
    raise ValueError(f"❌ Unsupported model: {model}")

def get_image_text_similarity(image_embedding: torch.Tensor, text: str, model: str = "CLIP") -> float:
    """
    Compute the similarity score between an image embedding and a text string.

    :param image_embedding: Torch tensor representing the image embedding.
    :param text: Text string to compare with the image.
    :param model: Name of the embedding model to use. Options are 'CLIP' or 'ALIGN'.
    :return: Cosine similarity score as a float.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "CLIP":
        # print("🔗 Computing similarity using CLIP model...")
        text_embedding = clip.get_text_embedding(text)
        return clip.compute_similarity(image_embedding, text_embedding)
    elif model == "ALIGN":
        # print("🔗 Computing similarity using ALIGN model...")
        text_embedding = align.get_text_embedding(text)
        return align.compute_similarity(image_embedding, text_embedding)
    
    raise ValueError(f"❌ Unsupported model: {model}")

if __name__ == "__main__":
    image_path = "sample_img.jpg"  # Replace with a valid test image path
    text_queries = [
        "a guy playing guitar",
        "a dog running in the park",
        "a bowl of fresh fruit"
    ]
    models = ["CLIP", "ALIGN"]

    # print("🧪 Starting image-text similarity tests...")

    for model_name in models:
        # print(f"\n🧠 Testing with model: {model_name}")
        try:
            image_embedding = get_image_embedding(image_path, model=model_name)
            assert isinstance(image_embedding, torch.Tensor), "Image embedding must be a torch.Tensor"
            # print(f"✅ Image embedding shape: {tuple(image_embedding.shape)}")

            for text in text_queries:
                score = get_image_text_similarity(image_embedding, text, model=model_name)
                assert isinstance(score, float), "Similarity score must be a float"
                # print(f"📌 Similarity with \"{text}\": {score:.4f}")

        except Exception as e:
            print(f"❌ Test failed for model {model_name}: {e}")

    # print("\n✅ All image-text tests completed.")  