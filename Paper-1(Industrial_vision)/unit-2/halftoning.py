import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Ordered Dithering (Bayer Matrix) ----------
def ordered_dither(image, matrix_size=4):
    # Generate Bayer matrix (normalized 0–255)
    def bayer_matrix(n):
        if n == 1:
            return np.array([[0]])
        else:
            prev = bayer_matrix(n // 2)
            tiles = [
                4 * prev + 0, 4 * prev + 2,
                4 * prev + 3, 4 * prev + 1
            ]
            return np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])
    
    bayer = bayer_matrix(matrix_size).astype(np.float32)
    bayer = (bayer + 0.5) / (matrix_size * matrix_size) * 255  # scale to [0,255]

    h, w = image.shape
    tiled = np.tile(bayer, (h // matrix_size + 1, w // matrix_size + 1))
    tiled = tiled[:h, :w]

    # Apply ordered dithering
    return (image > tiled).astype(np.uint8) * 255


# ---------- Floyd–Steinberg Error Diffusion ----------
def floyd_steinberg_dither(image):
    h, w = image.shape
    out = image.astype(np.float32).copy()

    for y in range(h):
        for x in range(w):
            old_pixel = out[y, x]
            new_pixel = 0 if old_pixel < 128 else 255
            out[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < w:
                out[y, x + 1] += error * 7 / 16
            if y + 1 < h and x > 0:
                out[y + 1, x - 1] += error * 3 / 16
            if y + 1 < h:
                out[y + 1, x] += error * 5 / 16
            if y + 1 < h and x + 1 < w:
                out[y + 1, x + 1] += error * 1 / 16

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- Main ----------
if __name__ == "__main__":
    # Load grayscale industrial image (update the path)
    # Using a standard, more accessible image for demonstration purposes
    # Create a gradient image if a path isn't available
    path = r"D:\HCL\archive\hazelnut\test\crack\000.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


    # Apply halftoning methods
    ordered = ordered_dither(img, matrix_size=4)
    fs = floyd_steinberg_dither(img)

    # Display results side by side
    titles = ["Original", "Ordered Dither (Bayer)", "Floyd-Steinberg Dither"]
    images = [img, ordered, fs]

    plt.figure(figsize=(15, 5)) # Adjusted size for better layout
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
        
    # Adjust layout to prevent titles from overlapping
    plt.tight_layout()

    output_filename = "dithering_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Combined image saved as '{output_filename}'")
    
    # Now, display the figure on screen
    plt.show()
