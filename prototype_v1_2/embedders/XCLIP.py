import torch
import numpy as np
import subprocess
import tempfile
from pathlib import Path

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

    def _load_frames(self, video_path, num_frames=8, tmp_dir="tmp_frames"):
        video_path = Path(video_path)
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # lấy thông tin video bằng ffprobe
        cmd_info = [
            "ffprobe", 
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets,r_frame_rate",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd_info, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"❌ Cannot probe video: {video_path}")
        
        lines = result.stdout.strip().split(",")
        if len(lines) < 2:
            raise RuntimeError(f"❌ Cannot parse ffprobe output: {result.stdout}")
        
        total_frames = int(lines[0])
        fps = lines[1]  # dạng "30/1" hoặc "25/1"
        
        if total_frames == 0:
            raise RuntimeError(f"❌ No frames found in video: {video_path}")
        if num_frames > total_frames:
            num_frames = total_frames

        # chọn index frame đều nhau
        positions = [
            int((i + 1) * total_frames / (num_frames + 1))
            for i in range(num_frames)
        ]

        frames = []
        for i, frame_idx in enumerate(positions):
            out_path = tmp_dir / f"frame_{i}.jpg"
            tmp_frame_idx = frame_idx
            while True:
                cmd_extract = [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-vf", f"select=eq(n\\,{tmp_frame_idx})",
                    "-vframes", "1",
                    "-q:v", "2",
                    str(out_path),
                    "-y"
                ]
                try:
                    subprocess.run(cmd_extract, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    img = Image.open(out_path).convert("RGB")
                    frames.append(img)
                    break
                except subprocess.CalledProcessError:
                    tmp_frame_idx += 1
                    if tmp_frame_idx >= total_frames:
                        raise RuntimeError(f"❌ Cannot extract frame {frame_idx} from video: {video_path}")
                    continue

        if len(frames) != num_frames:
            raise RuntimeError(
                f"fps: {fps}, total_frames: {total_frames}, "
                f"requested: {num_frames}, extracted: {len(frames)}"
            )
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