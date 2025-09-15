import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

class ALIGNWrapper:
    def __init__(self, model_name="kakaobrain/align-base", device=None):
        print("🚀 Initializing ALIGN model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_name).to(self.device)
        print(f"\t✅ ALIGN model loaded on {self.device}")

    def get_image_embedding(self, image_path):
        # print(f"\t🖼️  Loading image from {image_path}")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # print("🔍 Extracting image embedding...")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        # print("✅ Image embedding extracted.")
        return embedding

    def get_text_embedding(self, text):
        # print(f"\t✍️  Processing text: '{text}'")
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        # print("🔍 Extracting text embedding...")
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)

        # print("✅ Text embedding extracted.")
        return embedding

    def compute_similarity(self, image_embed, text_embed):
        # print("🔍 Computing similarity...")

        device = image_embed.device if image_embed.is_cuda else text_embed.device
        image_embed = image_embed.to(device)
        text_embed = text_embed.to(device)

        image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)

        similarity = torch.matmul(image_embed, text_embed.T).squeeze().item()
        # print(f"\t🎯 Similarity score: {similarity:.4f}")
        return similarity

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

    for text in text_candidates:
        align.compute_similarity(image_path, text)
