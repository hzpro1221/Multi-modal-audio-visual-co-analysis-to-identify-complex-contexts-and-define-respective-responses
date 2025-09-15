from semantic_video_info_extractor import semantic_video_info_extractor
from utils import fixed_interval_video_splitter
from pathlib import Path
import json
import os

if __name__ == "__main__":
    video_path = Path("./prototype_v1/KHCN01.mp4").resolve()
    chunk_infors = []
    questions = [
        "What is happening in the video?",
        "What are the key visuals and sounds?",
        "What message or purpose does the video convey?"
    ]

    print("==============================================")
    print("🔄 Step 1: Splitting video into chunks...")
    print("==============================================\n")

    output_dir = Path("./prototype_v1/video_chunks").resolve()
    os.makedirs(output_dir, exist_ok=True)
    interval_sec = 60

    video_chunks = fixed_interval_video_splitter(
        video_path=video_path,
        output_dir=output_dir,
        interval_sec=interval_sec,
        output_format="mp4",
        prefix="chunk"
    )

    print(f"✅ Done splitting video. Total chunks created: {len(video_chunks)}\n")

    print("==============================================")
    print("🎥 Step 2: Extracting semantic info from chunks")
    print("==============================================\n")

    for i, chunk_path in enumerate(video_chunks):
        chunk_path = Path(chunk_path).resolve()
        print(f"--- Chunk {i + 1}/{len(video_chunks)} ---")
        print(f"📂 Path: {chunk_path}")
        print("🔍 Extracting info...")

        segments_folder = chunk_path.with_suffix('').resolve()
        os.makedirs(segments_folder, exist_ok=True)

        results = semantic_video_info_extractor(
            video_path=chunk_path,
            segments_folder=segments_folder,
            questions=questions,
            transcribe=True,
            summary=True,
            answer_questions=True,
        )

        chunk_infors.append({
            "chunk_path": str(chunk_path),
            "results": results
        })

        print("✅ Done.\n")

    print("==============================================")
    print("💾 Step 3: Saving results to JSON file")
    print("==============================================\n")

    output_json = Path("./prototype_v1/chunk_infors.json").resolve()
    with open(output_json, "w") as f:
        json.dump(chunk_infors, f, ensure_ascii=False, indent=4)

    print(f"✅ Results saved to: {output_json}")
    print("🏁 All done!")
