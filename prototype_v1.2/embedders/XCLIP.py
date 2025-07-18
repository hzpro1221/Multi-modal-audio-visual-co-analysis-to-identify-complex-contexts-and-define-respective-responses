import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModel

class XCLIPWrapper:
    def __init__(self, model_name="microsoft/xclip-base-patch16", device=None, num_sampled_frames=8):
        print("🚀 Initializing X-CLIP model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_sampled_frames = num_sampled_frames
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"\t✅ X-CLIP loaded on {self.device} with {self.num_sampled_frames} sampled frames")

    def _load_frames(self, video_path):
        print(f"\t🎥 Loading and sampling frames from: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        N = self.num_sampled_frames
        frame_indices = [int(i * num_frames / N) for i in range(N)]
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [frame for frame in frames]
        print(f"\t✅ Sampled {len(frames)} frames")
        return frames

    def get_video_embedding(self, video_path):
        frames = self._load_frames(video_path)
        print("📊 Generating video embedding...")

        inputs = self.processor(
            videos=frames,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_video_features(**inputs)
            video_emb = outputs[0]  # shape: (dim,)
            video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)

        print("✅ Video embedding generated")
        return video_emb

    def get_text_embedding(self, text):
        print(f"\t📝 Generating text embedding for: '{text}'")
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_emb = outputs[0]  # shape: (dim,)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        print("✅ Text embedding generated")
        return text_emb

    def compute_similarity(self, video_embedding, text_embedding):
        print("🔍 Computing similarity...")

        device = video_embedding.device if video_embedding.is_cuda else text_embedding.device
        video_embedding = video_embedding.to(device)
        text_embedding = text_embedding.to(device)

        score = torch.matmul(video_embedding, text_embedding).item()
        print(f"\t🎯 Similarity score: {score:.4f}")
        return score

if __name__ == "__main__":
    wrapper = XCLIPWrapper(num_sampled_frames=8)

    video_path = "sample_video.mp4"
    texts = [
        "a cat running",
        "a dog running in the park",
        "a city skyline at night",
        "a football match"
    ]

    video_emb = wrapper.get_video_embedding(video_path)
    for text in texts:
        text_emb = wrapper.get_text_embedding(text)
        wrapper.compute_similarity(video_emb, text_emb)