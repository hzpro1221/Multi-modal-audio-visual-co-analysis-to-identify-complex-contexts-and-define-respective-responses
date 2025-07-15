import torch
import torch.nn.functional as F
import numpy as np
from pytorchvideo.models.hub import mvit_v2_s_16x1_detection
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, Resize, Normalize
from typing import List, Dict, Union
from pathlib import Path
import os
import urllib.request

# Use GPU if available, fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class HumanActionDetector:
    """
    A video-based human action detection module using MViTv2 pretrained on AVA.
    It detects human actors and recognizes their actions within temporal video clips.
    """

    def __init__(self, top_k: int = 1):
        """
        Initializes the MViT action detection model and loads AVA class labels.

        Args:
            top_k (int): Number of top action predictions to return for each detected person.
        """
        self.top_k = top_k

        # Load pretrained Multiscale Vision Transformer (MViTv2) model for action detection
        self.model = mvit_v2_s_16x1_detection(pretrained=True).eval().to(device)

        # Define preprocessing pipeline for input video frames
        self.transform = Compose([
            Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            Resize((256, 256)),  # Resize input frames
        ])

        # Load AVA action class names
        self.class_names = self._load_ava_classes()

    def _load_ava_classes(self) -> List[str]:
        """
        Loads the AVA action label names from the official text file.
        Downloads the file if it does not exist.

        Returns:
            List[str]: List of action label strings.
        """
        ava_path = Path("ava_action_list_v2.2.txt")
        if not ava_path.exists():
            url = "https://raw.githubusercontent.com/facebookresearch/pytorchvideo/main/MODEL_ZOO/AVA/ava_action_list_v2.2.txt"
            urllib.request.urlretrieve(url, ava_path)

        with open(ava_path, "r") as f:
            lines = f.readlines()
        # Format: "ID label", we extract the label only
        return [line.strip().split(" ", 1)[1] for line in lines]

    def detect(self, video_path: Union[str, Path], clip_duration: float = 8.0, stride: float = 4.0) -> List[Dict]:
        """
        Detects human actions throughout a video by applying MViT to sliding video clips.

        Args:
            video_path (Union[str, Path]): Path to the input video file.
            clip_duration (float): Duration (in seconds) of each analysis clip.
            stride (float): Sliding window step (in seconds) for overlapping clips.

        Returns:
            List[Dict]: List of action detection results, each including timestamp, bbox, action, and confidence.
        """
        video = EncodedVideo.from_path(str(video_path))
        results = []
        t = 0.0
        duration = video.duration

        # Loop over video using a sliding temporal window
        while t + clip_duration <= duration:
            # Extract clip of length clip_duration starting at time t
            frames = video.get_clip(t, t + clip_duration)["video"]  # Shape: (C, T, H, W)
            inputs = self.transform(frames).unsqueeze(0).to(device)  # Add batch dimension → (1, C, T, H, W)

            with torch.no_grad():
                # Run inference through MViT detection model
                output = self.model(inputs)["preds"][0]  # Get the first sample in batch

            boxes = output["boxes"].cpu().numpy()     # Person bounding boxes (normalized)
            scores = output["scores"].cpu().numpy()   # Action prediction confidence scores
            labels = output["labels"].cpu().numpy()   # Action class IDs (multi-label)

            # Iterate through each detected person
            for i in range(len(boxes)):
                person_box = boxes[i]
                person_scores = scores[i]
                person_labels = labels[i][:self.top_k]  # Top-K action labels

                # Append each detected action with metadata
                for j, label in enumerate(person_labels):
                    results.append({
                        "timestamp": {"start": t, "end": t + clip_duration},     # Approximate center time
                        "bbox": person_box.tolist(),                      # Normalized bounding box [x1, y1, x2, y2]
                        "action": self.class_names[label],                # Action class name
                        "confidence": float(person_scores[j])             # Confidence score for the action
                    })

            t += stride  # Slide window forward

        return results
