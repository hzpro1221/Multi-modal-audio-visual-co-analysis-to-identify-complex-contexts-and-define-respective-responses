import torch
from sentence_transformers import SentenceTransformer, util

class BGEWrapper:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device=None):
        print("🚀 Initializing BGE model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"\t✅ BGE model loaded on {self.device}")

    def get_text_embedding(self, text):
        # BGE performs better with prompt-style instructions
        prompt = f"Represent this sentence for searching relevant passages: {text}"
        with torch.no_grad():
            embedding = self.model.encode(prompt, convert_to_tensor=True, normalize_embeddings=False)
        return embedding

    def compute_similarity(self, topic_embed, query_embed):
        device = topic_embed.device if topic_embed.is_cuda else query_embed.device
        topic_embed = topic_embed.to(device)
        query_embed = query_embed.to(device)

        # Normalize embeddings
        topic_embed = topic_embed / topic_embed.norm(p=2, dim=-1, keepdim=True)
        query_embed = query_embed / query_embed.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = util.cos_sim(topic_embed, query_embed).item()
        return similarity

# Example test
if __name__ == "__main__":
    bge = BGEWrapper()

    query = "A cat resting on the sofa"
    candidates = [
        "A dog lying on the floor",
        "A cat curled up on the couch",
        "Children playing in the park",
    ]
    
    query_embed = bge.get_text_embedding(query)

    for text in candidates:
        topic_embed = bge.get_text_embedding(text)
        sim = bge.compute_similarity(topic_embed, query_embed)
        print(f"🔎 Similarity between:\n  '{query}'\n  '{text}'\n  → Score: {sim:.4f}\n")