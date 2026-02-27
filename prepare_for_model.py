import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def prepare_image(image_input):
    """
    Accepts:
    - File path (str)
    - numpy array
    - PIL Image (Gradio default)

    Returns:
    1) input_tensor for model
    2) rgb_image (0-1 normalized) for future Grad-CAM
    """

    # -----------------------------
    # Case 1: File path
    # -----------------------------
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError("Image could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Case 2: PIL Image (Gradio)
    # -----------------------------
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)

    # -----------------------------
    # Case 3: numpy array
    # -----------------------------
    elif isinstance(image_input, np.ndarray):
        image = image_input

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize to model size
    image_resized = cv2.resize(image, (224, 224))

    # Normalize for visualization (future Grad-CAM)
    rgb_image = image_resized.astype(np.float32) / 255.0

    # Transform for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image_resized).unsqueeze(0)

    return input_tensor, rgb_image
