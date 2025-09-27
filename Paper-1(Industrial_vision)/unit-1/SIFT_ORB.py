import cv2
import matplotlib.pyplot as plt
\
img = cv2.imread("brick_Wall.jpg", cv2.IMREAD_GRAYSCALE)

# -------- SIFT Keypoints --------
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img, None)
img_sift = cv2.drawKeypoints(img, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# -------- ORB Keypoints --------
orb = cv2.ORB_create(nfeatures=500)
kp_orb, des_orb = orb.detectAndCompute(img, None)
img_orb = cv2.drawKeypoints(img, kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title(f"SIFT Keypoints ({len(kp_sift)})")
plt.imshow(img_sift, cmap='gray')
plt.axis("off")
plt.subplot(1,2,2)
plt.title(f"ORB Keypoints ({len(kp_orb)})")
plt.imshow(img_orb, cmap='gray')
plt.axis("off")
plt.show()
