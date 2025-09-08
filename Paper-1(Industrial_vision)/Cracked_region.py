import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load the nut image ---
img = cv2.imread("D:/HCL/archive/tile/test/crack/000.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Denoise slightly
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Step 2: Edge detection (cracks are edges)
edges = cv2.Canny(blur, threshold1=50, threshold2=150)

# Step 3: Morphological operations to close small gaps
kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(edges, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 4: Create binary ground truth mask (0 = background, 255 = crack)
ground_truth = np.zeros_like(gray)
ground_truth[mask > 0] = 255

# --- Display ---
plt.figure(figsize=(12,5))

plt.subplot(1,4,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Gaussian image")
plt.imshow(blur)
plt.axis("off")


plt.subplot(1,4,3)
plt.title("Canny Edges")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.subplot(1,4,4)
plt.title("Ground Truth Mask (Binary)")
plt.imshow(ground_truth, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# --- Save Ground Truth Mask ---
cv2.imwrite("ground_truth_mask.png", ground_truth)
