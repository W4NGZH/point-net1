import cv2

img = cv2.imread('D:/img/data927/paper_used/065.png')
dst = img[0:384,60:412]
gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
cv2.imwrite('D:/img/data927/paper_used/cj065.png',gray)