import json
import os

def replace_special_chars(text, mapping):
    for old_char, new_char in mapping.items():
        text = text.replace(old_char, new_char)
    return text

if __name__ == "__main__":
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    MAPPING = {
        "？": "?",
        "｜": "|",
        "：": ":",
        "’": "'",
        "＜": "<",
        "＞": ">",
    }
    for item in metadata:
        try: 
            old_video_path = item["video_path"]
            old_chapter_path = item["chapter_path"]  
            
            item["video_name"] = replace_special_chars(item["video_name"], MAPPING)
            item["chapter_name"] = replace_special_chars(item["chapter_name"], MAPPING)
            item["video_path"] = replace_special_chars(item["video_path"], MAPPING)
            item["chapter_path"] = replace_special_chars(item["chapter_path"], MAPPING)

            os.rename(old_video_path, item["video_path"])
            os.rename(old_chapter_path, item["chapter_path"])
        except FileNotFoundError:
            print(f"File not found: {old_video_path} or {old_chapter_path}")

    filenames = os.listdir(".")
    for file in filenames:
        try:
            old_video_path = file
            new_video_path = replace_special_chars(file, MAPPING)

            os.rename(old_video_path, new_video_path)
        except FileNotFoundError:
            print(f"File not found: {old_video_path}")

    filenames = os.listdir(".")
    for sample in dataset:
        try:
            if (sample["chapter_title"] is not None):
                vid_path = [vid for vid in metadata if vid.get("video_name") == sample["video_title"] and vid.get("chapter_name") == sample["chapter_title"]][0].get("chapter_path")
            else:
                vid_path = ([vid for vid in metadata if vid.get("video_name") == sample["video_title"]] + [{"video_path": file} for file in filenames if (sample["video_title"] in file)])[0].get("video_path")
            sample["video_path"] = os.path.basename(vid_path)
        except IndexError:
            print(f"Metadata not found for: {sample['video_title']} - {sample['chapter_title']}")

    with open("dataset.json", 'w') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
