import cv2
import matplotlib.pyplot as plt

# Read image using OpenCV
img_cv = cv2.imread('red.jpg')

# Check shape
print("OpenCV Image Shape:", img_cv.shape)

# Convert BGR to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Resize image
img_resized = cv2.resize(img_cv, (300, 300))

# Convert to grayscale
img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Save resized image
cv2.imwrite('opencv_resized.jpg', img_resized)

# Display all images
plt.figure(figsize=(10, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Resized image (300x300)
plt.subplot(2, 2, 2)
plt.imshow(img_resized)  # Convert for display
plt.title('Resized Image')
plt.axis('off')

# Grayscale image
plt.subplot(2, 2, 3)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.show()
