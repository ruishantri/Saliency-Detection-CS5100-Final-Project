import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

root = os.getcwd()
imgPath = os.path.join(r'sample_image_bird.jpg')
image = cv.imread(imgPath)

imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

figure = plt.figure()
rows, cols = 2, 5

figure.add_subplot(rows, cols, 1)
plt.title('gray version')
plt.axis("off")
plt.imshow(gray)

# find an approximate estimate of the image
ret, thresh = cv.threshold(gray, thresh=0, maxval=255,
                           type=cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
figure.add_subplot(rows, cols, 2)
plt.title('initial extraction')
plt.axis("off")
plt.imshow(thresh)

# remove noise
kernel = np.ones((3, 3), dtype=np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
figure.add_subplot(rows, cols, 3)
plt.title('without noise')
plt.axis("off")
plt.imshow(opening)

# obtain sure background
sure_bg = cv.dilate(opening, kernel, iterations=3)
figure.add_subplot(rows, cols, 4)
plt.title('sure background')
plt.axis("off")
plt.imshow(sure_bg)

# obtain sure foreground
dist_transform = cv.distanceTransform(
    opening, distanceType=cv.DIST_L2, maskSize=5)  # heat map
figure.add_subplot(rows, cols, 5)
plt.title('heat map')
plt.axis("off")
plt.imshow(dist_transform)


ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
figure.add_subplot(rows, cols, 6)
plt.title('sure foreground')
plt.axis("off")
plt.imshow(sure_fg)

# obtain unknown area
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
# unknown = cv.subtract(thresh, sure_fg)  # without having sure_bg
figure.add_subplot(rows, cols, 7)
plt.title('unknown area')
plt.axis("off")
plt.imshow(unknown)

# marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# mark the region of unknown with zero
markers[unknown == 255] = 0
figure.add_subplot(rows, cols, 8)
plt.title('marker labelling')
plt.axis("off")
plt.imshow(markers)


markers = np.int32(markers)
markers = cv.watershed(imgRGB, markers)
figure.add_subplot(rows, cols, 9)
plt.title('after calling watershed')
plt.axis("off")
plt.imshow(markers)

# change the intensities of the edges and draw boundaries in red
imgRGB[markers == -1] = [255, 0, 0]
figure.add_subplot(rows, cols, 10)
plt.title('final result')
plt.axis("off")
plt.imshow(imgRGB)

figure.suptitle('Watershed Algorithm', x=0.5, y=0.05)
plt.show()
