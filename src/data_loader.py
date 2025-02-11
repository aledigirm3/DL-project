import os
from glob import glob

def load_video_paths_and_labels(dataset_folder):
    """
    Scans the folder and collects the video paths with automatic labels.
    """
    video_paths = []
    labels = []

    all_videos = glob(os.path.join(dataset_folder, "*.mp4"))

    for video in all_videos:
        filename = os.path.basename(video)

        if filename.startswith("bachata"):
            labels.append(1)  # 1 = Bachata
        elif filename.startswith("salsa"):
            labels.append(0)  # 0 = Salsa
        else:
            continue

        video_paths.append(video)


    print(f"🔹 {len(video_paths)} Videos found")
    print(f"🕺 Bachata: {labels.count(1)} - Salsa: {labels.count(0)}\n")
    return video_paths, labels
