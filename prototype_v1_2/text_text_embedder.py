from embedders.BGE import BGEWrapper  # Make sure this path matches your project

import torch
from torch import Tensor

# Initialize BGE-3 model
bge = BGEWrapper(model_name="BAAI/bge-large-en-v1.5")

def get_text_embedding(text: str, model: str = "BGE-3") -> torch.Tensor:
    """
    Generate a text embedding using the specified model.

    :param text: The input text to embed.
    :param model: Name of the embedding model to use. Currently only 'BGE-3'.
    :return: Torch tensor representing the text embedding.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "BGE-3":
        return bge.get_text_embedding(text)
    
    raise ValueError(f"❌ Unsupported model: {model}")
    
def get_text_text_similarity(topic_embed: Tensor, query: str, model: str = "BGE-3") -> float:
    """
    Compute the similarity score between a precomputed topic embedding and a query text.

    :param topic_embed: Precomputed torch.Tensor embedding for the topic text.
    :param query: The input query text string to compare.
    :param model: Name of the embedding model to use. Currently only 'BGE-3'.
    :return: Cosine similarity score as a float.
    :raises ValueError: If an unsupported model is provided.
    """
    model = model.upper()
    if model == "BGE-3":
        query_embed = bge.get_text_embedding(query)
        return bge.compute_similarity(topic_embed, query_embed)
    
    raise ValueError(f"❌ Unsupported model: {model}")

if __name__ == "__main__":
    # Define topic and queries
    topic = "A child playing with a ball in the garden"
    queries = [
        "A kid having fun outdoors",
        "A dog sleeping in the house",
        "Children playing in the playground",
        "A person cooking in the kitchen"
    ]

    # Generate topic embedding once
    topic_embed = get_text_embedding(topic)

    # Compute and print similarity with each query
    for query in queries:
        score = get_text_text_similarity(topic_embed, query)
        print(f"🔎 Similarity between:\n  Topic: \"{topic}\"\n  Query: \"{query}\"\n  → Score: {score:.4f}\n")
