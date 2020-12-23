from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np
import cv2

a = io.imread('ambassadors.jpg')
skull = a[800:1243, 357:780]
fig = plt.figure()
# plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side

ax1.imshow(a)
ax2.imshow(skull)

plt.show()


# 利用 getPerspectiveTransform 來線性轉換
def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    # 算出轉換攻勢
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title("Original Image", fontsize=30)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title("Distortion Correction Result", fontsize=30)
        plt.show()
    else:
        return warped, M


w, h = skull.shape[0], skull.shape[1]
# We will first manually select the source points
# we will select the destination point which will map the source points in
# original image to destination points in unwarped image
# 手動找座標(因為還沒教怎麼自動找)
src = np.float32([(0, 277),
                  (0, 180),
                  (450, 100),
                  (450, 0)])

dst = np.float32([
    (505, 100),
    (300, 40),
    (100, 350),
    (10, 200),
])

unwarp(skull, src, dst, True)
