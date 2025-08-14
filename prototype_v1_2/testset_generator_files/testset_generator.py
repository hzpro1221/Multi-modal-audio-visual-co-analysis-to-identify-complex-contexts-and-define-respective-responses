import json

def slide_paragraph(text, window_size):
    words = text.split()
    if (len(words) <= window_size):
        return [text]
    else:
        return [' '.join(words[i:i+window_size]) for i in range(len(words) - window_size + 1)]

if __name__ == "__main__":
    with open("testset_metadata.json", "r") as file:
        testset_metadata = json.load(file)

    for cluster in testset_metadata:
        for video in cluster["videos"]:
            video["chunks"] = slide_paragraph(
                text=video["presentation_script"],
                window_size=35
            )

    with open("testset_metadata1.json", "w", encoding="utf-8") as f:
        json.dump(testset_metadata, f, indent=2, ensure_ascii=False)




