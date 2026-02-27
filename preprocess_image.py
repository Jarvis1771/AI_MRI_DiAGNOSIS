import cv2
import numpy as np

# Load grayscale image
image = cv2.imread("data/raw/sample_scan.png", cv2.IMREAD_GRAYSCALE)

# Resize image (standard for AI models)
image = cv2.resize(image, (224, 224))

# Apply CLAHE (contrast improvement)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image = clahe.apply(image)

# Normalize image
image = image / 255.0

print("Preprocessing completed successfully")

# Show processed image
cv2.imshow("Preprocessed CT/MRI", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
