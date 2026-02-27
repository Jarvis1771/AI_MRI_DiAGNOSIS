import cv2

# Load image in grayscale
image = cv2.imread("data/raw/sample_scan.png", cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if image is None:
    print("Image not found. Check file path.")
else:
    print("Image loaded successfully")

    # Show image
    cv2.imshow("CT / MRI Scan", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
