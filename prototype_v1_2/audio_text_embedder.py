from embedders.CLAP import CLAPModelWrapper
import torch
from torch import Tensor

# Initialize the CLAP model wrapper
clap = CLAPModelWrapper()

def get_audio_embeddings(audio_path: str, model: str = "CLAP") -> Tensor:
    """
    Generate an audio embedding using the specified model.

    :param audio_path: Path to the audio file (e.g., ".wav" or ".mp3").
    :param model: Name of the embedding model to use. Only 'CLAP' is supported.
    :return: PyTorch tensor representing the audio embedding.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "CLAP":
        print("🔊 Using CLAP model for audio embedding.")
        return clap.get_audio_embedding(audio_path)
    
    raise ValueError(f"❌ Unsupported model: {model}")

def get_audio_text_similarity(audio_embedding: Tensor, text: str, model: str = "CLAP") -> float:
    """
    Compute the similarity score between an audio embedding and a text string.

    :param audio_embedding: PyTorch tensor representing the audio embedding.
    :param text: Text string to compare with the audio.
    :param model: Name of the embedding model to use. Only 'CLAP' is supported.
    :return: Cosine similarity score as a float.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "CLAP":
        # print("🔗 Computing similarity using CLAP model...")
        text_embedding = clap.get_text_embedding(text)
        return clap.compute_similarity(audio_embedding, text_embedding)
    
    raise ValueError(f"❌ Unsupported model: {model}")

if __name__ == "__main__":
    # Setup test
    audio_path = "sample_audio.mp3"  # Replace with your actual test audio path
    text_queries = [
        "a dog barking",
        "a cat meowing",
        "a person talking"
    ]

    # print("🎬 Starting CLAP embedding test...")

    try:
        # Get audio embedding
        audio_embedding = get_audio_embeddings(audio_path, model="CLAP")
        assert isinstance(audio_embedding, Tensor), "Audio embedding must be a torch.Tensor"
        # print("✅ Audio embedding shape:", audio_embedding.shape)

        # Compute similarity for each text
        for text in text_queries:
            score = get_audio_text_similarity(audio_embedding, text, model="CLAP")
            assert isinstance(score, float), "Similarity score must be a float"
            # print(f"📌 Similarity with \"{text}\": {score:.4f}")

        # print("✅ All tests passed.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
