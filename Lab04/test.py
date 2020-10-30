import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgb2gray
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk, ball
from skimage.filters import rank

c = io.imread('Lenna.jpg')
io.imshow(c)
grayscaleImage = rgb2gray(c)

import skimage.exposure as ex

gamma2Pic = ex.adjust_gamma(grayscaleImage, 3)

height, width = gamma2Pic.shape
EqualizedImage = np.empty_like(gamma2Pic)
EqualizedImage[:] = gamma2Pic * 255
EqualizedImage = EqualizedImage.astype(int)
numCounter = {}

for i in range(height):
    for j in range(width):
        if EqualizedImage[i, j] not in numCounter:
            numCounter[EqualizedImage[i, j]] = 1
        else:
            numCounter[EqualizedImage[i, j]] += 1

cdf = []

cdf.append(0 if 0 not in numCounter else numCounter.get(0))
for i in range(1, 256):
    if i in numCounter:
        cdf.append(cdf[i - 1] + numCounter.get(i))
    else:
        cdf.append(cdf[i - 1])

cdfMax = max(cdf)
cdfMin = min(cdf)

for i in range(height):
    for j in range(width):
        EqualizedImage[i][j] = ((cdf[EqualizedImage[i][j]] - cdfMin) / (cdfMax - cdfMin)) * 255

io.imshow(EqualizedImage)
f = plt.figure()
f.show(plt.hist(EqualizedImage.flatten(), bins=256))
