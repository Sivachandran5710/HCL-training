import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load Image ---
img = cv2.imread("006.png", cv2.IMREAD_GRAYSCALE)

# --- Define High-Pass Filter Kernel ---
hpf_kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

# --- Apply High-Pass Filter ---
high_pass = cv2.filter2D(img, -1, hpf_kernel)

# --- Enhance Details (Add back to Original Image) ---
enhanced = cv2.add(img, high_pass)

# --- Display Results ---
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")  # corrected title
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("High-Pass Filtered")
plt.imshow(high_pass, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Enhanced Details")
plt.imshow(enhanced, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
