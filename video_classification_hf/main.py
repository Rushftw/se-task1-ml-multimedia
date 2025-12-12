import sys
from pathlib import Path

from transformers import pipeline

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_file>")
        return

    video_path = Path(sys.argv[1])
    if not video_path.is_file():
        print(f"File not found: {video_path}")
        return

    print(f"Loading video-classification pipeline with model {MODEL_NAME} ...")
    video_pipe = pipeline(
        task="video-classification",
        model=MODEL_NAME
    )

    print(f"Running inference on {video_path} ...")
    results = video_pipe(str(video_path))

    print("\nPredictions:")
    for r in results:
        print(f"{r['label']}: {r['score']:.4f}")


if __name__ == "__main__":
    main()
