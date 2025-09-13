import torch
import numpy as np
import subprocess
import tempfile
import os
import cv2
from PIL import Image

from transformers import AutoProcessor, AutoModel

class XCLIPWrapper:
    def __init__(self, model_name="microsoft/xclip-base-patch16", device=None, num_sampled_frames=8):
        print("🚀 Initializing X-CLIP model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_sampled_frames = num_sampled_frames
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"\t✅ X-CLIP loaded on {self.device} with {self.num_sampled_frames} sampled frames")

    def _load_frames(self, video_path, num_frames=8):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if num_frames > total_frames:
            num_frames = total_frames

        timestamps = [(i + 1) * duration / (num_frames + 1) for i in range(num_frames)]

        frames = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)  # seek theo mili giây
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(img)

        cap.release()

        if len(frames) != num_frames:
            raise RuntimeError(f"❌ Extracted {len(frames)} frames, expected {num_frames}")

        return frames
    def get_video_embedding(self, video_path):
        frames = self._load_frames(video_path)
        # print("📊 Generating video embedding...")

        inputs = self.processor(
            videos=frames,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_video_features(**inputs)
            video_emb = outputs[0]  # shape: (dim,)

        # print("✅ Video embedding generated")
        return video_emb

    def get_text_embedding(self, text):
        # print(f"\t📝 Generating text embedding for: '{text}'")
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_emb = outputs[0]  # shape: (dim,)
        # print("✅ Text embedding generated")
        return text_emb

    def compute_similarity(self, video_embedding, text_embedding):
        # print("🔍 Computing similarity...")

        device = video_embedding.device if video_embedding.is_cuda else text_embedding.device
        video_embedding = video_embedding.to(device)
        text_embedding = text_embedding.to(device)

        video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)           

        score = torch.matmul(video_embedding, text_embedding).item()
        # print(f"\t🎯 Similarity score: {score:.4f}")
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