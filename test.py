import imutils
import numpy as np
import cv2
import cv2 as cv
from matplotlib import pyplot as plt

lic_data = cv2.CascadeClassifier('./output-hv-33-x25.xml')


def plt_show(image, title="", gray=False, size=(100, 100)):
    temp = image
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(temp, cmap='gray')
        plt.show()


def detect_number(img):
    temp = img
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    number = lic_data.detectMultiScale(img, 1.2)
    print("number plate detected:"+str(len(number)))
    for numbers in number:
        (x, y, w, h) = numbers
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+h]
        cv2.rectangle(temp, (x, y), (x+w, y+h), (0, 255, 0), 3)

    plt_show(temp)


img = cv2.imread("./xemay2.jpg")
plt_show(img)
detect_number(img)
plt.subplot(1, 1, 1), plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('input image', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

plt.subplot(1, 1, 1), plt.imshow(erosion)
plt.title(''), plt.xticks([]), plt.yticks([])
plt.show()
