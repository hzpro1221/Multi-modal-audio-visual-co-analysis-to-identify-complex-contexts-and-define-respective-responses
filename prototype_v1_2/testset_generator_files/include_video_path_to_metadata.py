import json

if __name__ == "__main__":
    with open("testset_metadata1.json", "r") as file:
        testset_metadata = json.load(file)

    for cluster in testset_metadata:
        for video in cluster["videos"]:
            video["video_path"] = f"video/{video['video_name']}.mp4"

    with open("testset_metadata2.json", "w", encoding="utf-8") as f:
        json.dump(testset_metadata, f, indent=2, ensure_ascii=False)

