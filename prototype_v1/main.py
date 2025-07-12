from semantic_video_info_extractor import semantic_video_info_extractor
from utils import fixed_interval_video_splitter

import json

if __name__ == "__main__":
    video_path = "prototype_v1/KHCN01.mp4"
    chunk_infors = []
    questions = [
        "What is happening in the video?",
        "What are the key visuals and sounds?",
        "What message or purpose does the video convey?"
    ]

    # Spliting the video into chunks
    output_dir = "prototype_v1/video_chunks"
    interval_sec = 60  # Split every 10 seconds
    video_chunks = fixed_interval_video_splitter(
        video_path=video_path,
        output_dir=output_dir,
        interval_sec=interval_sec,
        output_format="mp4",
        prefix="chunk"
    )

    for chunk_path in video_chunks:
        print(f"Chunk created: {chunk_path}")
        results = semantic_video_info_extractor(
            video_path=video_path,
            questions=questions,
            transcribe=True,
            summary=True,
            human_actions=True,
            answer_questions=True
        )
        chunk_infors.append(results)

    with open("prototype_v1/chunk_infors", "w") as f:
        json.dump(chunk_infors, f, indent=4)   
