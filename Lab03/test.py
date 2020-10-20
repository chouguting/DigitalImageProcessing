import cv2
import numpy as np

theImage = cv2.imread("Lenna.jpg")
# 先把彩色圖片轉成黑白的
theImageGray = cv2.cvtColor(theImage, cv2.COLOR_BGR2GRAY)
# 秀出黑白圖片
cv2.imshow("show photo", theImageGray)
# 等按按鈕後關掉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

DitherArray = np.array([[0.513, 0.272, 0.724, 0.483, 0.543, 0.302, 0.694, 0.453],
                        [0.151, 0.755, 0.091, 0.966, 0.181, 0.758, 0.121, 0.936],
                        [0.634, 0.392, 0.574, 0.332, 0.664, 0.423, 0.604, 0.362],
                        [0.060, 0.875, 0.211, 0.815, 0.030, 0.906, 0.241, 0.845],
                        [0.543, 0.302, 0.694, 0.453, 0.513, 0.272, 0.724, 0.483],
                        [0.181, 0.758, 0.121, 0.936, 0.151, 0.755, 0.091, 0.936],
                        [0.664, 0.423, 0.604, 0.362, 0.634, 0.392, 0.574, 0.332],
                        [0.030, 0.906, 0.241, 0.845, 0.060, 0.875, 0.211, 0.815]]) * 255

theDitheredImage = theImageGray
width, height = theDitheredImage.shape
for i in range(0, height, 8):
    for j in range(0, width, 8):
        theDitheredImage[i:i + 8, j:j + 8] = (DitherArray > theDitheredImage[i:i + 8, j:j + 8]) * 255

# 秀出黑白圖片
cv2.imshow("show photo", theDitheredImage)
# 等按按鈕後關掉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
