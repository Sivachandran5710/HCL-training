import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('006.png', cv2.IMREAD_GRAYSCALE)

# --- Sobel Edge Detection ---
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x direction
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y direction
sobel = np.sqrt(sobelx**2 + sobely**2)

# --- Prewitt Edge Detection ---
prewitt_kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
prewitt_kernel_y = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

prewittx = cv2.filter2D(img, -1, prewitt_kernel_x)
prewitty = cv2.filter2D(img, -1, prewitt_kernel_y)
prewitt = np.sqrt(prewittx.astype(float)**2 + prewitty.astype(float)**2)

# --- Plot Results ---
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')

plt.show()
