import torch
import torchaudio
from transformers import ClapProcessor, ClapModel

class CLAPModelWrapper:
    """
    Wrapper for the HuggingFace CLAP model to compute audio-text similarity.
    Supports 10-second audio chunking and tracks progress with logs.
    """

    def __init__(self, model_name="laion/clap-htsat-fused", device=None):
        print("🚀 Initializing HuggingFace CLAP model...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = 48000
        self.chunk_size = self.sampling_rate * 10  # 10 seconds

        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        print(f"\t✅ CLAP model loaded on {self.device}")

    def _load_audio(self, file_path):
        # print(f"\t🎧 Loading audio from: {file_path}")
        waveform, sr = torchaudio.load(file_path)

        if sr != self.sampling_rate:
            # print(f"\t🔄 Resampling from {sr} Hz to {self.sampling_rate} Hz...")
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            # print("🔉 Converting stereo to mono...")
            waveform = waveform.mean(dim=0)

        return waveform

    def _chunk_audio(self, waveform):
        total_samples = waveform.shape[0]
        chunks = []
        for i in range(0, total_samples, self.chunk_size):
            chunk = waveform[i: i + self.chunk_size]
            chunks.append(chunk.numpy().astype("float32"))
        # print(f"\t🔪 Audio chunked into {len(chunks)} segment(s) of 10s")
        return chunks

    def get_audio_embedding(self, audio_path):
        """
        Returns a single averaged embedding over all 10s audio chunks.
        """
        waveform = self._load_audio(audio_path)
        chunks = self._chunk_audio(waveform)

        # print("🔎 Extracting audio embedding...")
        inputs = self.processor(
            audios=chunks,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            audio_embeds = self.model.get_audio_features(**inputs)

        avg_audio_embedding = audio_embeds.mean(dim=0, keepdim=True)
        # print("✅ Audio embedding generated.")
        return avg_audio_embedding  # shape: [1, dim]

    def get_text_embedding(self, text):
        """
        Returns a single text embedding.
        """
        # print(f"\t📝 Generating text embedding for: '{text}'")
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_embed = self.model.get_text_features(**inputs)

        # print("✅ Text embedding generated.")
        return text_embed[0].unsqueeze(0)  # shape: [1, dim]

    def compute_similarity(self, audio_embedding, text_embedding):
        # print("🔍 Computing similarity...")

        device = audio_embedding.device if audio_embedding.is_cuda else text_embedding.device
        audio_embedding = audio_embedding.to(device)
        text_embedding = text_embedding.to(device)

        audio_embedding = audio_embedding / audio_embedding.norm(p=2, dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)        

        score = torch.matmul(audio_embedding, text_embedding.T).item()
        # print(f"\t🎯 Similarity score: {score:.4f}")
        return score

if __name__ == "__main__":
    clap = CLAPModelWrapper()

    audio_path = "sample_audio.mp3"
    text_candidates = ["a cat meowing", "a dog barking", "a person speaking"]

    audio_emb = clap.get_audio_embedding(audio_path)

    for text in text_candidates:
        text_emb = clap.get_text_embedding(text)
        clap.compute_similarity(audio_emb, text_emb)
