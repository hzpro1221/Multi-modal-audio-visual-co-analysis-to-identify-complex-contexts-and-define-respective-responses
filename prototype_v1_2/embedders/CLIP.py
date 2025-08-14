import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPWrapper:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        print("🚀 Initializing CLIP model...")
        self.device = device or ("cuda:1" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        print(f"\t✅ CLIP loaded on {self.device}")

    def get_image_embedding(self, image_path):
        # print(f"\t🖼 Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # print("📊 Generating image embedding...")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            image_emb = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        # print("✅ Image embedding generated")
        return image_emb

    def get_text_embedding(self, text):
        # print(f"\t📝 Generating text embedding for: '{text}'")
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_emb = outputs[0] / outputs[0].norm(p=2, dim=0, keepdim=True)
        # print("✅ Text embedding generated")
        return text_emb

    def compute_similarity(self, image_emb, text_emb):
        # print("🔍 Computing similarity...")

        device = image_emb.device if image_emb.is_cuda else text_emb.device
        image_emb = image_emb.to(device)
        text_emb = text_emb.to(device)

        score = torch.matmul(image_emb.unsqueeze(0), text_emb.unsqueeze(1)).item()
        # print(f"\t🎯 Similarity score: {score:.4f}")
        return score

if __name__ == "__main__":
    clip = CLIPWrapper()

    image_path = "sample_img.jpg"
    texts = [
        "a tree",
        "a cat",
        "a man playing guitar",
        "a sunset over the ocean"
    ]

    image_emb = clip.get_image_embedding(image_path)
    for text in texts:
        text_emb = clip.get_text_embedding(text)
        clip.compute_similarity(image_emb, text_emb)
