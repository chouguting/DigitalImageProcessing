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

# Dot Diffusion
ClassMatrix = np.array([[204, 0, 5, 33, 51, 59, 23, 118, 54, 69, 40, 160, 169, 110, 168, 188],
                        [3, 6, 22, 36, 60, 50, 74, 115, 140, 82, 147, 164, 171, 142, 220, 214],
                        [14, 7, 42, 16, 63, 52, 94, 56, 133, 152, 158, 177, 179, 208, 222, 1],
                        [15, 26, 43, 75, 79, 84, 148, 81, 139, 136, 166, 102, 217, 219, 226, 4],
                        [17, 39, 72, 92, 103, 108, 150, 135, 157, 193, 190, 100, 223, 225, 227, 13],
                        [28, 111, 99, 87, 116, 131, 155, 112, 183, 196, 181, 224, 232, 228, 12, 21],
                        [47, 120, 91, 105, 125, 132, 172, 180, 184, 205, 175, 233, 245, 8, 20, 41],
                        [76, 65, 129, 137, 165, 145, 178, 194, 206, 170, 229, 244, 246, 19, 24, 49],
                        [80, 73, 106, 138, 176, 182, 174, 197, 218, 235, 242, 249, 247, 18, 48, 68],
                        [101, 107, 134, 153, 185, 163, 202, 173, 231, 241, 248, 253, 44, 88, 70, 45],
                        [123, 141, 149, 61, 195, 200, 221, 234, 240, 243, 254, 38, 46, 77, 104, 109],
                        [85, 96, 156, 130, 203, 215, 230, 250, 251, 252, 255, 53, 62, 93, 86, 117],
                        [151, 167, 189, 207, 201, 216, 236, 239, 25, 31, 34, 113, 83, 95, 124, 114],
                        [144, 146, 191, 209, 213, 237, 238, 29, 32, 55, 64, 97, 126, 78, 128, 159],
                        [187, 192, 198, 212, 9, 10, 30, 35, 58, 67, 90, 71, 122, 127, 154, 161],
                        [199, 210, 211, 2, 11, 27, 37, 57, 66, 89, 98, 121, 119, 143, 162, 186]])

ErrorArray = np.array([[0.38459, 1, 0.38459],
                       [1, 0, 1],
                       [0.38459, 1, 0.38459]])

theDotDiffusedImage = theImageGray
height, width = theDotDiffusedImage.shape
tempClassArray = np.zeros((16 + 2, 16 + 2))
tempClassArray[1:16 + 1, 1:16 + 1] = ClassMatrix

for i in range(0, height, 16):
    for j in range(0, width, 16):
        tempArray = np.zeros((16 + 2, 16 + 2))
        tempArray[1:16 + 1, 1:16 + 1] = theDotDiffusedImage[i:i + 16, j:j + 16]
        for findNumber in range(0, 256):
            tempErrorArray = np.empty_like(ErrorArray)
            tempErrorArray[:] = ErrorArray
            result = np.where(ClassMatrix == findNumber)
            place = list(zip(result[0], result[1]))
            old = tempArray[1 + place[0][0], 1 + place[0][1]]
            new = (old // 127) * 255
            tempArray[1 + place[0][0], 1 + place[0][1]] = new
            E = old - new
            sum = 0
            for k in range(-1, 2):
                for m in range(-1, 2):
                    if tempClassArray[1 + place[0][0] + k, 1 + place[0][1] + m] <= tempClassArray[
                        1 + place[0][0], 1 + place[0][1]]:
                        tempErrorArray[k + 1, m + 1] = 0;
                    sum += tempErrorArray[k + 1, m + 1]
            tempErrorArray = tempErrorArray / sum
            tempArray[1 + place[0][0] - 1:1 + place[0][0] + 2, 1 + place[0][1] - 1:1 + place[0][1] + 2] = tempArray[
                                                                                                          1 + place[0][
                                                                                                              0] - 1:1 +
                                                                                                                     place[
                                                                                                                         0][
                                                                                                                         0] + 2,
                                                                                                          1 + place[0][
                                                                                                              1] - 1:1 +
                                                                                                                     place[
                                                                                                                         0][
                                                                                                                         1] + 2] + E * tempErrorArray
        theDotDiffusedImage[i:i + 16, j:j + 16] = tempArray[1:16 + 1, 1:16 + 1]

# 秀出圖片
cv2.imshow("theDotDiffusedImage", theDotDiffusedImage)
# 等按按鈕後關掉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
