import json

def present_generate_prompt(cluster_name, video_name):
    topic = f"{cluster_name} - {video_name}"
    prompt = f"""
You are a presenter. You task is to prepare a presentation script for this following topic: {topic}
Please your answer as a continuous paragraph.
"""
    return prompt

if __name__ == "__main__":
    test_data = [
            {
                "cluster_name": "Foundations and Definition of AI",
                "videos": [
                    {
                        "video_name": "What is AI?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is Machine Learning?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is Deep Learning?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is Generative AI?",
                        "presentation_script": "" 
                    },
                ]
            },
            {
                "cluster_name": "AI Learning Methods",
                "videos": [
                    {
                        "video_name": "What is Supervised Learning?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is Unsupervised Learning?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is Semi-supervised Learning?",
                        "presentation_script": "" 
                    },
                ]
            },        
            {
                "cluster_name": "AI Architectures and Main Models",
                "videos": [
                    {
                        "video_name": "What is Transformer Architecture?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is K-Nearest Neighbors (KNN) Algorithm?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is Neural Networks?",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "What is Recurrent Neural Networks?",
                        "presentation_script": "" 
                    },                
                ]
            },
            {
                "cluster_name": "Applications of AI",
                "videos": [
                    {
                        "video_name": "Birthday check",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "Translate language",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "Car navigation",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "Car Assembly",
                        "presentation_script": "" 
                    },
                    {
                        "video_name": "Crop Inspection",
                        "presentation_script": "" 
                    },                                                
                ]
            },
    ]

    for cluster in test_data:
        for video in cluster["videos"]:
            print(f"\nCluster - {cluster['cluster_name']}; Video - {video['video_name']}")
            
            prompt = present_generate_prompt(
                cluster_name=cluster['cluster_name'], 
                video_name=video['video_name']
            )

            print(f"Prompt:{prompt}")
            presentation_script = input("Presentation script:")
            video["presentation_script"] = presentation_script

    with open("testset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)



