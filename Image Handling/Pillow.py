from PIL import Image
import matplotlib.pyplot as plt
# Read image with Pillow
img_pil = Image.open('red.jpg')

# Check size
print("Pillow Image Size:", img_pil.size)

# Resize image
img_pil_resized = img_pil.resize((300, 300))

# Convert to grayscale
img_pil_gray = img_pil.convert('L')

# Save resized image
img_pil_resized.save('pillow_resized.jpg')

plt.figure(figsize=(10, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(img_pil)
plt.title('Original Image')
plt.axis('off')

# Resized image (300x300)
plt.subplot(2, 2, 2)
plt.imshow(img_pil_resized)  # Convert for display
plt.title('Resized Image')
plt.axis('off')

# Grayscale image
plt.subplot(2, 2, 3)
plt.imshow(img_pil_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.show()
