import numpy as np
import cv2

import numpy as np
import cv2


import cv2
import numpy as np

import cv2
import numpy as np

import cv2
import numpy as np

def remove_snow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return result


# Example usage
src_img = cv2.imread('/home/oem/桌面/drone/snow.jpeg')
snow_removed = remove_snow(src_img)

# Display the result
cv2.imshow('Original Image', src_img)
cv2.imshow('Dehazed Image', snow_removed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image = cv2.imread('/home/oem/桌面/drone/fog.jpeg')
# image = Transmission(image)
# cv2.imshow('Image Window', image)
# cv2.waitKey(0)