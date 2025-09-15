import json
import os
import ffmpeg

from dotenv import load_dotenv
from google.cloud import storage
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.json_format import MessageToDict

from uuid import uuid4
unique_id = str(uuid4())

# === Load biến môi trường ===
load_dotenv()

GCP_PROJECT_ID = None
GCP_REGION = "us-central1"
GCP_BUCKET_NAME = "audio_transcript1231342"
GCP_GCS_PATH = f"../gcs_audio/audio_{unique_id}.mp3" # You may want to make this dynamic
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = None

def transcribe_audio(audio_path):
    print("🔊 Uploading to GCS and transcribing via Google STT...")

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(GCP_GCS_PATH)
    blob.upload_from_filename(audio_path)
    gcs_uri = f"gs://{GCP_BUCKET_NAME}/{GCP_GCS_PATH}"
    print(f"✔ Uploaded to: {gcs_uri}")

    # Prepare STT client
    client = SpeechClient(client_options=ClientOptions(api_endpoint=f"{GCP_REGION}-speech.googleapis.com"))
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["vi-VN"],
        model="chirp",
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        ),
    )
    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/recognizers/_",
        config=config,
        files=[{"uri": gcs_uri}],
        recognition_output_config={"inline_response_config": {}}
    )

    # Transcribe
    operation = client.batch_recognize(request=request)
    response = operation.result()
    response_dict = MessageToDict(response._pb)

    # Extract text
    results = response_dict.get("results", {}).get(gcs_uri, {}).get("transcript", {}).get("results", [])
    full_text = []
    for result in results:
        for alt in result.get("alternatives", []):
            full_text.append(alt.get("transcript", ""))

    final_text = "\n".join(full_text).strip()
    
    # Xóa file trên GCS
    try:
        blob.delete()
        print(f"🗑 Deleted from GCS: {gcs_uri}")
    except Exception as e:
        print(f"⚠ Failed to delete {gcs_uri}: {e}")    
    return final_text


def extract_audio(video_path, output_audio_path):
    ffmpeg.input(video_path).output(output_audio_path, acodec='mp3', ac=1).run(overwrite_output=True)
    print("Audio extracted.")
    return output_audio_path

if __name__ == "__main__":
    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    for sample in dataset:
        try:
            video = os.path.join("video", sample['video_path'])

            base_name = os.path.splitext(video)[0]
            print(base_name)
            audio_path = base_name + "_denoise.mp3"
            extract_audio(video, audio_path)
            transcribe_text = transcribe_audio(audio_path=audio_path)

            sample["script"] = transcribe_text
        except:
            sample["script"] = False
            continue

    with open("dataset_transcript.json", "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)         
