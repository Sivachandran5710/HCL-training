import cv2
import matplotlib.pyplot as plt

# Load image (grayscale for clarity)
img = cv2.imread("D:/HCL/archive/screw/test/good/001.png", cv2.IMREAD_GRAYSCALE)

# Box Filter (Average Blurring)
box_filtered = cv2.blur(img, (5,5))   # 5x5 kernel

# Gaussian Filter
gaussian_filtered = cv2.GaussianBlur(img, (5,5), 1)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(box_filtered, cmap='gray')
plt.title("Box Filter (Mean Blurring)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title("Gaussian Filter (Edge Preserving)")
plt.axis('off')

plt.show()
