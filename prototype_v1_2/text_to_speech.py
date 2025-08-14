from TTS.api import TTS
import os 

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
def text_to_speech(text, output_path):
    tts.tts_to_file(text=text, file_path=output_path)

if __name__ == "__main__":
    sample_text = "Artificial Intelligence (AI) refers to the simulation of human intelligence by machines, especially computer systems. It enables computers to perform tasks that typically require human cognition, such as learning, reasoning, problem-solving, perception, and language understanding. With advancements in machine learning, deep learning, and natural language processing, AI is increasingly integrated into daily life—from virtual assistants and recommendation systems to autonomous vehicles and medical diagnostics. As AI continues to evolve, it holds the potential to transform industries, enhance productivity, and solve complex global challenges, while also raising important ethical and societal questions."
    output_file = "input.wav"

    print("Generating speech...")
    text_to_speech(sample_text, output_file)

    if os.path.exists(output_file):
        print(f"✅ Audio successfully saved to: {output_file}")
    else:
        print("❌ Failed to generate audio.")