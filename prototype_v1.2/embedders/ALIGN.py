import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

class ALIGNWrapper:
    def __init__(self, model_name="kakaobrain/align-base", device=None):
        print("🚀 Initializing ALIGN model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_name).to(self.device)
        print(f"✅ ALIGN model loaded on {self.device}")

    def get_image_embedding(self, image_path):
        print(f"🖼️  Loading image from {image_path}")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        print("🔍 Extracting image embedding...")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)

        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        print("✅ Image embedding extracted.")
        return embedding

    def get_text_embeddings(self, text_list):
        print(f"✍️  Processing {len(text_list)} text candidates...")
        inputs = self.processor(text=text_list, padding=True, return_tensors="pt").to(self.device)

        print("🔍 Extracting text embeddings...")
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)

        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        print("✅ Text embeddings extracted.")
        return embeddings

    def compute_similarity(self, image_path, text_list):
        print("🔗 Computing similarity between image and text...")
        image_embed = self.get_image_embedding(image_path)
        text_embeds = self.get_text_embeddings(text_list)

        similarities = torch.matmul(image_embed, text_embeds.T).squeeze(0)
        scores = similarities.cpu().tolist()

        print("📊 Similarity scores:")
        for text, score in zip(text_list, scores):
            print(f"   - '{text}': {score:.4f}")
        return scores

# Example test
if __name__ == "__main__":
    align = ALIGNWrapper()

    image_path = "sample_img.jpg"
    text_candidates = [
        "a tree",
        "a cat",
        "a man playing guitar",
        "a sunset over the ocean"
    ]

    align.compute_similarity(image_path, text_candidates)
