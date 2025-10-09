import cv2
import matplotlib.pyplot as plt

# ----------------------
# Load Image
# ----------------------
image = cv2.imread("fov_input.png") 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Original image dimensions
height, width, _ = image.shape

# Example: 24mm, 35mm, 50mm, 85mm, 135mm (simulated)
focal_lengths = [24, 35, 50, 85, 135]
base_focal = 24  # Treat the original image as base focal length

def crop_for_fov(img, base_focal, target_focal):
    """
    Crops the image to simulate narrower FOV for longer focal lengths.
    """
    scale = base_focal / target_focal  # scale < 1 for zoom in
    h, w, _ = img.shape
    
    new_w, new_h = int(w * scale), int(h * scale)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    # Resize back to original size for consistent visualization
    return cv2.resize(cropped, (w, h))

# ----------------------
# Apply crop for each focal length
# ----------------------
plt.figure(figsize=(15,5))
for i, f in enumerate(focal_lengths):
    sim_image = crop_for_fov(image, base_focal, f)
    plt.subplot(1, len(focal_lengths), i+1)
    plt.imshow(sim_image)
    plt.title(f"{f}mm")
    plt.axis('off')

plt.suptitle("Simulated FOV Changes with Different Focal Lengths")
plt.show()
