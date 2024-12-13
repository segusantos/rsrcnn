import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Load the original image
rotations = []
for i in range(0,4):
    temp = cv2.imread(f'../data/train/ONEIMG/X2_P420_S420/GT/patch_{i}.bmp')
    rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    rotations.append(rgb)

# add %10, %20, %30, %40 white padding to each image
rotations9 = [cv2.copyMakeBorder(img, int(img.shape[0]*0.1), int(img.shape[0]*0.1), int(img.shape[1]*0.1), int(img.shape[1]*0.1), cv2.BORDER_CONSTANT, value=[255, 255, 255]) for img in rotations]
rotations8 = [cv2.copyMakeBorder(img, int(img.shape[0]*0.2), int(img.shape[0]*0.2), int(img.shape[1]*0.2), int(img.shape[1]*0.2), cv2.BORDER_CONSTANT, value=[255, 255, 255]) for img in rotations]
rotations7 = [cv2.copyMakeBorder(img, int(img.shape[0]*0.3), int(img.shape[0]*0.3), int(img.shape[1]*0.3), int(img.shape[1]*0.3), cv2.BORDER_CONSTANT, value=[255, 255, 255]) for img in rotations]
rotations6 = [cv2.copyMakeBorder(img, int(img.shape[0]*0.4), int(img.shape[0]*0.4), int(img.shape[1]*0.4), int(img.shape[1]*0.4), cv2.BORDER_CONSTANT, value=[255, 255, 255]) for img in rotations]

# plot each set of images
fig, axs = plt.subplots(5, 4, figsize=(15, 15))

for i, img in enumerate(rotations):
    axs[0, i].imshow(img)
    axs[0, i].axis('off')
    # axs[0, i].set_title(f"Rotation {i*90}°", fontsize=8)

for i, img in enumerate(rotations9):
    axs[1, i].imshow(img)
    axs[1, i].axis('off')
    # axs[1, i].set_title(f"Rotation {i*90}°", fontsize=8)

for i, img in enumerate(rotations8):
    axs[2, i].imshow(img)
    axs[2, i].axis('off')
    # axs[2, i].set_title(f"Rotation {i*90}°", fontsize=8)

for i, img in enumerate(rotations7):
    axs[3, i].imshow(img)
    axs[3, i].axis('off')
    # axs[3, i].set_title(f"Rotation {i*90}°", fontsize=8)

for i, img in enumerate(rotations6):
    axs[4, i].imshow(img)
    axs[4, i].axis('off')
    # axs[4, i].set_title(f"Rotation {i*90}°", fontsize=8)




plt.tight_layout()
plt.savefig('./plotScripts/rotations.png')
