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
        print(f"\t✅ CLIP4Clip loaded on {self.device} - sampling {self.num_frames} frames per video")

    def _sample_frames(self, container):
        # print("🖼 Sampling frames from video...")
        total = container.streams.video[0].frames
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == self.num_frames:
                break
        # print(f"\t✅ Sampled {len(frames)} frames.")
        return frames

    def get_video_embedding(self, video_path):
        # print(f"\t📥 Processing video: {video_path}")
        container = av.open(video_path)
        frames = self._sample_frames(container)
        # print("📊 Encoding visual frames...")
        vision_inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        vision_emb = self.vision_model(**vision_inputs).image_embeds
        # print("✅ Video embedding generated.")
        return vision_emb

    def get_text_embedding(self, text):
        # print(f"\t📝 Processing text: '{text}'")
        text_inputs = self.text_tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_emb = self.text_model(**text_inputs).text_embeds[0]
        # print("✅ Text embedding generated.")
        return text_emb

    def compute_similarity(self, video_emb, text_emb):
        # print("🔍 Computing similarity...")

        device = video_emb.device if video_emb.is_cuda else text_emb.device
        video_emb = video_emb.to(device)
        text_emb = text_emb.to(device)

        video_emb = video_emb / video_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)        

        score = torch.matmul(video_emb, text_emb.T).item()
        return score

if __name__ == "__main__":
    wrapper = Clip4ClipWrapper(num_frames=8)

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