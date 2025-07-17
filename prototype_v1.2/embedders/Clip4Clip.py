import torch
import av
import numpy as np
from transformers import AutoProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

class Clip4ClipWrapper:
    def __init__(self, model_name="Searchium-ai/clip4clip-webvid150k", device=None, num_frames=8):
        print("🚀 Initializing CLIP4Clip model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.num_frames = num_frames
        print(f"✅ CLIP4Clip loaded on {self.device} – sampling {self.num_frames} frames per video")

    def _sample_frames(self, container):
        print("🖼 Sampling frames from video...")
        total = container.streams.video[0].frames
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == self.num_frames:
                break
        print(f"✅ Sampled {len(frames)} frames.")
        return frames

    def get_video_embedding(self, video_path):
        print(f"📥 Processing video: {video_path}")
        container = av.open(video_path)
        frames = self._sample_frames(container)
        print("📊 Encoding visual frames...")
        vision_inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        vision_emb = self.vision_model(**vision_inputs).vision_model_output
        vision_emb = vision_emb / vision_emb.norm(dim=-1, keepdim=True)
        print("✅ Video embedding generated.")
        return vision_emb

    def get_text_embedding(self, text):
        print(f"📝 Processing text: '{text}'")
        text_inputs = self.text_tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_emb = self.text_model(**text_inputs).text_embeds[0]
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        print("✅ Text embedding generated.")
        return text_emb

    def compute_similarity(self, video_emb, text_emb):
        print("🔍 Computing similarity...")
        sim = torch.matmul(video_emb, text_emb.T)
        score = sim.item()
        print(f"🎯 Similarity score: {score:.4f}")
        return score
