import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load and Prepare Image ---
try:
    image_bgr = cv2.imread('peppers.png')
    if image_bgr is None:
        raise FileNotFoundError
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
except FileNotFoundError:
    print("Error: 'peppers.png' not found. Please download it or replace with your image path.")
    # As a fallback, create a simple color image
    image_rgb = np.zeros((300, 400, 3), dtype=np.uint8)
    image_rgb[50:150, 50:150] = [255, 0, 0]    # Red square
    image_rgb[50:150, 200:300] = [0, 255, 0]  # Green square
    image_rgb[170:270, 125:225] = [0, 0, 255]  # Blue square

# --- 2. Convert to HSV and LAB Color Spaces ---
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

# Split channels for visualization
h, s, v = cv2.split(image_hsv)
l, a, b = cv2.split(image_lab)

# --- 3. Visualize the Color Spaces and Their Channels ---
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
plt.suptitle("Color Space Conversions and Channels", fontsize=16)

# Row 1: Original and RGB Channels
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original RGB')
axes[0, 1].imshow(image_rgb[:,:,0], cmap='gray')
axes[0, 1].set_title('Red Channel')
axes[0, 2].imshow(image_rgb[:,:,1], cmap='gray')
axes[0, 2].set_title('Green Channel')
axes[0, 3].imshow(image_rgb[:,:,2], cmap='gray')
axes[0, 3].set_title('Blue Channel')

# Row 2: HSV Channels
axes[1, 0].imshow(image_hsv)
axes[1, 0].set_title('HSV (Incorrect Display)')
axes[1, 1].imshow(h, cmap='gray')
axes[1, 1].set_title('Hue (H)')
axes[1, 2].imshow(s, cmap='gray')
axes[1, 2].set_title('Saturation (S)')
axes[1, 3].imshow(v, cmap='gray')
axes[1, 3].set_title('Value (V)')

# Row 3: LAB Channels
axes[2, 0].imshow(image_lab)
axes[2, 0].set_title('LAB (Incorrect Display)')
axes[2, 1].imshow(l, cmap='gray')
axes[2, 1].set_title('Lightness (L*)')
axes[2, 2].imshow(a, cmap='gray')
axes[2, 2].set_title('a* (green-red)')
axes[2, 3].imshow(b, cmap='gray')
axes[2, 3].set_title('b* (blue-yellow)')

for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the first figure
plt.savefig("color_space_channels.png", bbox_inches='tight', dpi=150)
print("Saved color space analysis to 'color_space_channels.png'")
plt.show()


# --- 4. Assess Segmentation using HSV ---
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)

lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)

red_mask = mask1 + mask2
segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=red_mask)

# --- 5. Display and SAVE Segmentation Results ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.suptitle("Segmentation Assessment using HSV Color Space", fontsize=16)

axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(red_mask, cmap='gray')
axes[1].set_title('Binary Mask for Red Color')
axes[1].axis('off')

axes[2].imshow(segmented_image)
axes[2].set_title('Segmented Red Peppers')
axes[2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.92])

# Save the second (final) figure
plt.savefig("segmentation_result.png", bbox_inches='tight', dpi=150)
print("Saved segmentation result to 'segmentation_result.png'")
plt.show()