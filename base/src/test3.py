import cv2

# Path to input image
image_path = "../../images/cube/capture25.jpg"

# Load image
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Get image dimensions
h, w = img.shape[:2]

# Ellipse parameters
cx, cy = int((w // 2)*1.05), int(h * 0.47)
axes = (int(w * 0.2), int(h * 0.25))

# Draw ellipse on image
cv2.ellipse(
    img,                # image
    (cx, cy),           # center
    axes,               # radii
    0,                  # rotation angle
    0,                  # start angle
    360,                # end angle
    (0, 255, 0),        # green color (BGR)
    3                   # thickness
)

# Save result
output_path = "ellipse_overlay.jpg"
cv2.imwrite(output_path, img)

print(f"Saved to: {output_path}")