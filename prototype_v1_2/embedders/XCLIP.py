import torch
import numpy as np
from pathlib import Path
from PIL import Image

from transformers import AutoProcessor, AutoModel

import decord
from decord import VideoReader, cpu

class XCLIPWrapper:
    def __init__(self, model_name="microsoft/xclip-base-patch16", device=None, num_sampled_frames=8):
        print("🚀 Initializing X-CLIP model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_sampled_frames = num_sampled_frames
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"\t✅ X-CLIP loaded on {self.device} with {self.num_sampled_frames} sampled frames")

    def _probe_video_info(self, video_path):
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        return total_frames, fps

    def _load_frames(self, video_path, num_frames=8):
        video_path = Path(video_path)
        total_frames, fps = self._probe_video_info(video_path)

        if total_frames == 0:
            raise RuntimeError(f"❌ No frames found in video: {video_path}")
        if num_frames > total_frames:
            num_frames = total_frames

        # chọn index frame đều nhau
        positions = [
            int((i + 1) * total_frames / (num_frames + 1))
            for i in range(num_frames)
        ]

        vr = VideoReader(str(video_path), ctx=cpu(0))
        frames = []

        for idx in positions:
            tmp_idx = idx
            while True:
                try:
                    frame = vr[tmp_idx].asnumpy()  # decord trả ndarray
                    img = Image.fromarray(frame).convert("RGB")
                    frames.append(img)
                    break
                except Exception as e:
                    tmp_idx += 1
                    if tmp_idx >= total_frames:
                        raise RuntimeError(f"❌ Cannot extract frame {idx} from video: {video_path}") from e
                    continue

        if len(frames) != num_frames:
            raise RuntimeError(
                f"fps: {fps}, total_frames: {total_frames}, "
                f"requested: {num_frames}, extracted: {len(frames)}"
            )
        return frames
    
    def get_video_embedding(self, video_path):
        frames = self._load_frames(video_path, num_frames=self.num_sampled_frames)

        inputs = self.processor(
            videos=frames,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_video_features(**inputs)
            video_emb = outputs[0]  # shape: (dim,)

        return video_emb

    def get_text_embedding(self, text):
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_emb = outputs[0]  # shape: (dim,)

        return text_emb

    def compute_similarity(self, video_embedding, text_embedding):
        device = video_embedding.device if video_embedding.is_cuda else text_embedding.device
        video_embedding = video_embedding.to(device)
        text_embedding = text_embedding.to(device)

        video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)           

        score = torch.matmul(video_embedding, text_embedding).item()
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
        score = wrapper.compute_similarity(video_emb, text_emb)
        print(f"Text: '{text}' | Similarity: {score:.4f}")
