import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Load the original image
original_image = cv2.imread('../data/train/ONEIMG/X2_P420_S420/GT/patch_1.bmp')
# # Convert the image to RGB (OpenCV loads images in BGR by default)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# # Get the image dimensions
height, width, _ = original_image.shape

# Define the 4 parts (corners of the image)
top_left = original_image_rgb[:height//2, :width//2]
top_right = original_image_rgb[:height//2, width//2:]
bottom_left = original_image_rgb[height//2:, :width//2]
bottom_right = original_image_rgb[height//2:, width//2:]

# Create the plot
fig = plt.figure(figsize=(12, 6))

# Plot the original image on the left
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(original_image_rgb)
ax1.set_title("Original Image", fontsize=12)
ax1.axis('off')

# Add an arrow between the original image and the corners
arrow = FancyArrowPatch((0.4, 0.5), (0.6, 0.5), transform=fig.transFigure, 
                         arrowstyle="->", mutation_scale=20, color="black", lw=2)
fig.add_artist(arrow)

# Create a 2x2 grid for the corner images
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis('off')
ax2.set_title("Corners", fontsize=12)

# Combine and plot the corners
fig_corners, axs = plt.subplots(2, 2, figsize=(6, 6))

for (ax, corner, title) in zip(axs.flat, 
                               [top_left, top_right, bottom_left, bottom_right], 
                               ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]):
    ax.imshow(corner)
    # ax.set_title(title, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig('./plotScripts/image_corners.png')


image_corners = cv2.imread('./plotScripts/image_corners.png')
# change to BRG
image_corners_rgb = cv2.cvtColor(image_corners, cv2.COLOR_BGR2RGB)

# plot original (an arrow) corners

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Assume original_image_rgb and image_corners are already defined
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a single figure with 2 subplots

# Plot the original image on the left
axes[0].imshow(original_image_rgb)
# axes[0].set_title("Original Image", fontsize=12)
axes[0].axis('off')

# Plot the corners image on the right
axes[1].imshow(image_corners_rgb)
# axes[1].set_title("Corners", fontsize=12)
axes[1].axis('off')

# Add an arrow between the subplots
arrow = FancyArrowPatch((0.46, 0.5), (0.56, 0.5), transform=fig.transFigure, 
                          mutation_scale=70, color="black", lw=7)
fig.add_artist(arrow)

# Adjust layout and show
plt.tight_layout()
plt.savefig('./plotScripts/original_and_corners.png')
