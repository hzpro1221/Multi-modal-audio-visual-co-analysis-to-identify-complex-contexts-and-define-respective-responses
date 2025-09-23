from utils import extract_audio_from_video, extract_keyframe_from_video
from embedders.CLIP import CLIPWrapper
from embedders.CLAP import CLAPModelWrapper
from embedders.XCLIP import XCLIPWrapper

import tempfile
import os
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkRetriever(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, video_representer_layer=8, text_representer_layer=8):
        super(NeuralNetworkRetriever, self).__init__()
        self.audio_embedder = CLAPModelWrapper()
        self.image_embedder = CLIPWrapper()
        self.video_embedder = XCLIPWrapper()

        # Frozen parameter
        self.audio_embedder.model.requires_grad_(False)
        self.image_embedder.model.requires_grad_(False)
        self.video_embedder.model.requires_grad_(False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation="gelu",
            batch_first=True
        )
        self.video_representer = nn.TransformerDecoder(decoder_layer, num_layers=video_representer_layer).to(self.device)
        self.text_representer = nn.TransformerDecoder(decoder_layer, num_layers=text_representer_layer).to(self.device)

    def _apply_pos_rope(self, embed, base=10000):
        # embed: (seq_len, dim) or (batch, seq_len, dim)
        if embed.dim() == 2:
            embed = embed.unsqueeze(0)  # (1, seq_len, dim)

        batch_size, seq_len, dim = embed.shape
        half_dim = dim // 2

        freq_seq = torch.arange(half_dim, dtype=embed.dtype, device=embed.device)
        inv_freq = 1.0 / (base ** (freq_seq / half_dim))

        pos = torch.arange(seq_len, device=embed.device).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        angle = pos * inv_freq.unsqueeze(0).unsqueeze(0)  # (1, seq_len, half_dim)

        sin, cos = angle.sin(), angle.cos()

        x1 = embed[:, :, :half_dim]
        x2 = embed[:, :, half_dim:]

        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated  # (batch, seq_len, dim)

    def _video_combine_embeds(self, video_paths):
      audio_paths = []
      keyframe_paths_list = []
      tmpdir = "/content/tmpdir"
      for idx, video_path in enumerate(video_paths):
        print(f"video_path: {video_path}")
        audio_path = os.path.join(tmpdir, f"audio_{idx}.wav")
        keyframe_dir = os.path.join(tmpdir, f"keyframes_{idx}")
        os.makedirs(keyframe_dir, exist_ok=True)

        audio_path = extract_audio_from_video(
            video_path=video_path,
            output_path=audio_path
        )

        try:
            keyframe_paths = extract_keyframe_from_video(
                video_path=video_path,
                num_frame=16,
                output_folder=keyframe_dir
            )
        except subprocess.CalledProcessError as e:
            print(f"Error extracting keyframes from {video_path}: {e}")
            print(f"Command: {e.cmd}")
            print(f"Stderr: {e.stderr.decode()}")
            raise # Re-raise the exception after printing debug info

        audio_paths.append(audio_path)
        keyframe_paths_list.append(keyframe_paths)

      print(f"keyframe_paths: {keyframe_paths}")
      print(f"audio_paths: {audio_paths}")
      print(f"video_paths: {video_paths}")
      # Expect lists of equal length
      audio_embeddings = [self.audio_embedder.get_audio_embedding(p) for p in audio_paths]
      image_embeddings = [self.image_embedder.get_image_embedding(kp) for kp in keyframe_paths_list]
      video_embeddings = [self.video_embedder.get_video_embedding(vp) for vp in video_paths]

      # LayerNorm
      ln = nn.LayerNorm(audio_embeddings[0].shape[-1]).to(self.device)

      combined = []
      for a, i, v in zip(audio_embeddings, image_embeddings, video_embeddings):
          a = ln(a)
          i = ln(i)
          v = ln(v.unsqueeze(0))   # (1, dim)
          combined.append(torch.cat([a, i, v], dim=0))  # (3, dim)

      combined_embed = torch.stack(combined, dim=0)  # (B, 3, dim)
      return combined_embed

    def _text_combine_embeds(self, texts):
        audio_embeddings = [self.audio_embedder.get_text_embedding(t) for t in texts]
        image_embeddings = [self.image_embedder.get_text_embedding(t) for t in texts]
        video_embeddings = [self.video_embedder.get_text_embedding(t) for t in texts]

        # LayerNorm (chỉ khởi tạo 1 lần)
        ln = nn.LayerNorm(audio_embeddings[0].shape[-1]).to(self.device)

        combined = []
        for a, i, v in zip(audio_embeddings, image_embeddings, video_embeddings):
            a = ln(a)             # (1, dim)
            i = ln(i.unsqueeze(0))  # (1, dim)
            v = ln(v.unsqueeze(0))  # (1, dim)

            combined.append(torch.cat([a, i, v], dim=0))  # (3, dim)

        combined_embed = torch.stack(combined, dim=0)  # (B, 3, dim)
        return combined_embed

    def get_video_embed(self, input_embeds):
        # input_embeds: (B, num_candidates, seq_len, dim)
        B, C, S, D = input_embeds.shape
        input_embeds = input_embeds.view(B * C, S, D)

        tgt = torch.zeros(B * C, 1, D, device=self.device)
        rep_token = self.video_representer(tgt=tgt, memory=input_embeds)  # (B*C, 1, D)

        rep_token = rep_token.squeeze(1).view(B, C, D)  # (B, num_candidates, D)
        return rep_token

    def get_text_embed(self, input_embeds):
        # input_embeds: (B, seq_len, dim)

        B, seq_len, dim = input_embeds.shape
        tgt = torch.zeros(B, 1, dim, device=self.device)

        rep_token = self.text_representer(tgt=tgt, memory=input_embeds)  # (B, 1, dim)
        return rep_token.squeeze(1)  # (B, dim)

    def compute_similarity(self, text_embeds, video_embeds):
        print(f"text_embeds.shape: {text_embeds.shape}")
        print(f"video_embeds.shape: {video_embeds.shape}")
        text_norm = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)       # (B, dim)
        video_norm = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)    # (B, C, dim)

        print(f"text_norm.shape: {text_norm.shape}")
        print(f"video_norm.shape: {video_norm.shape}")

        sims = torch.bmm(
            video_norm,                   # (B, C, dim)
            text_norm.unsqueeze(-1)       # (B, dim, 1)
        ).squeeze(-1)                     # (B, C)

        return sims
# if __name__ == "__main__":
#     print("Load model...")
#     model = NeuralNetworkRetriever().to("cuda" if torch.cuda.is_available() else "cpu")
#     model.load_state_dict(torch.load("/content/model.pth"))

#     MAX_TEXT_LEN = 70
#     EVALUATE_FOLDER = "/content/Multi-modal-audio-visual-co-analysis-to-identify-complex-contexts-and-define-respective-responses/prototype_v1_2/evaluate_videos"
#     GET_VIDEO_EMBEDDER = model._video_combine_embeds
#     GET_TEXT_EMBEDDER =  model._text_combine_embeds

#     video_paths = list(Path(EVALUATE_FOLDER).rglob("*.mp4"))
#     video_embeds = {}
#     print("Preprocess video...")

#     for video_path in video_paths:
#         if video_path not in video_embeds:
#             emb = GET_VIDEO_EMBEDDER([video_path]).squeeze(0)
#             video_embeds[video_path] = emb
#     print(video_embeds)
#     while True:
#         query = input("What is your query? (type 'exit' to quit): ")
#         if query.lower() == 'exit':
#             break

#         if len(query) > MAX_TEXT_LEN:
#             print(f"Query too long! Max length is {MAX_TEXT_LEN} characters.")
#             continue

#         print(f"GET_TEXT_EMBEDDER([query]).shape: {GET_TEXT_EMBEDDER([query]).shape}")
#         print(f"GET_TEXT_EMBEDDER([query]).squeeze(0).shape: {GET_TEXT_EMBEDDER([query]).squeeze(0).shape}")
#         text_embed_output = model.get_text_embed(GET_TEXT_EMBEDDER([query]))
#         scores = {}
#         candidate_vids = torch.cat([video_embed.unsqueeze(0) for video_path, video_embed in video_embeds.items()])
#         video_embeds_output = model.get_video_embed(candidate_vids.unsqueeze(0))

#         print(f"text_embed_output.shape: {text_embed_output.shape}")
#         print(f"video_embeds_output.shape: {video_embeds_output.shape}")
#         sim = model.compute_similarity(text_embed_output, video_embeds_output)
#         topk_values, topk_indices = sim.topk(3, dim=-1)  # along candidate dim

#         keys = list(video_embeds.keys())
#         for i, (score, idx) in enumerate(zip(topk_values[0], topk_indices[0])):
#             video_path = keys[idx]
#             filename = os.path.basename(video_path)
#             print(f"Video: {filename}, Similarity Score: {score.item():.4f}")