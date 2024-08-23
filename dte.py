import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the pre-trained model from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def detect_objects(image_path):
    # Load the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the bounding boxes and labels
    boxes = predictions[0]['boxes'].numpy()
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()

    return boxes, labels, scores

def draw_boxes(image_path, boxes, labels, scores, threshold=0.5):
    image = cv2.imread(image_path)
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python detect.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    boxes, labels, scores = detect_objects(image_path)
    result_image = draw_boxes(image_path, boxes, labels, scores)

    # Save the result image
    result_image_path = "result_" + image_path
    cv2.imwrite(result_image_path, result_image)
    print(f"Result image saved as {result_image_path}")
