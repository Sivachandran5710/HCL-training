import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load initial image
img = cv2.imread('lotus.jpg') 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to get binary mask
_, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Define kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Apply opening then closing
mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title("Initial Image")
plt.imshow(img_rgb)
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Binary Mask")
plt.imshow(mask, cmap='gray')
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Cleaned Mask")
plt.imshow(mask_clean, cmap='gray')
plt.axis("off")

plt.show()
