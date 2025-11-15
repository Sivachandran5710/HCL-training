import cv2
import matplotlib.pyplot as plt

img = cv2.imread("006.png", cv2.IMREAD_GRAYSCALE)

#Box Filter
avg_filtered = cv2.blur(img, (5,5))   # 5x5 kernel

# Gaussian Filter
gauss_filtered = cv2.GaussianBlur(img, (5,5), sigmaX=1)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Scanned Document")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Box Filter")
plt.imshow(avg_filtered, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Gaussian Filter")
plt.imshow(gauss_filtered, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
