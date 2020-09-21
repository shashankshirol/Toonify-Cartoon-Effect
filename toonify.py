import cv2 as cv
import numpy as np
import sys
import math

# Helper Functions

def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    return edged

def Quantize_colors(img, a=24):
    img = np.floor_divide(img, a)
    img = img*a
    img = img.astype(np.uint8)
    return img

#Code:

name = 'Tiger.jpg'
img = cv.imread(name)

if(len(sys.argv) > 1):
    scale = int(sys.argv[1])
    w = int(img.shape[0] * scale / 100)
    h = int(img.shape[1] * scale / 100)

    img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)

# Applying median filtering to remove any salt and pepper noise present
img = cv.medianBlur(img, 7)

# Applying the Canny Edge detection
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = auto_canny(gray_img)

# Applying Dilation to thicken and smoothen the contours
kernel = np.ones((2, 2), np.uint8)
gray_img = cv.dilate(gray_img, kernel, iterations=1)

# Applying Bilateral filtering to obtain Cartooning Effect
for i in range(14):
    img = cv.bilateralFilter(img, 9, 17, 17)

# Quantizing colors to enhance the cartooning Effect
img = Quantize_colors(img)

# Merging the edges and the original image
img[gray_img==255] = [0, 0, 0]

# Cartoonized image
cv.imshow('Cartoonized Image', img)
cv.waitKey()

cv.imwrite('cartoonized_'+name, img)

cv.destroyAllWindows()
