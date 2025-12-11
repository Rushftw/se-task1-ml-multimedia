import sys
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request


def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as f:
        labels = [line.strip().decode("utf-8") for line in f.readlines()]
    return labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_file>")
        return

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"File not found: {image_path}")
        return

    print("Loading ResNet-18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        output = model(input_batch)

    probs = torch.nn.functional.softmax(output[0], dim=0)

    labels = load_labels()
    top5_prob, top5_catid = torch.topk(probs, 5)

    print("\nTop 5 predictions:")
    for i in range(top5_prob.size(0)):
        label = labels[top5_catid[i]]
        score = top5_prob[i].item()
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
