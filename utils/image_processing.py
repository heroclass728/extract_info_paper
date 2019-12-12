import cv2


class ImageProcessing:

    def __init__(self):
        pass

    def process_image(self, image):
        th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
        return th2
