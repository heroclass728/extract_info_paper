import cv2
import numpy as np


kernel2 = np.ones((2, 2), np.uint8)
THRESH_AREA = 5


class DetectDigits:

    def __init__(self, path, model):

        self.dirpath = path
        self.model = model

    def detect_handwritten_digits(self, image):

        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        width = image.shape[1]
        # Convert to grayscale and apply Gaussian filtering
        # im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # Threshold the image
        # thresh_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
        thresh_image_not = cv2.bitwise_not(image)

        # Find contours in the image
        contours, _ = cv2.findContours(thresh_image_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        correct_contours = extract_correct_contour(contours)
        rects = [cv2.boundingRect(ctr) for ctr in correct_contours]
        # for rect in rects:
        #     cv2.rectangle(image_bgr, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0,255),1)
        #     cv2.imshow("image_bgr", image_bgr)
        #     cv2.waitKey()

        total_value = ""

        if len(rects) == 1 and rects[0][2] > 0.7 * width:

            img1 = thresh_image_not[:, :int(width/2)]
            img1_gray = cv2.bitwise_not(img1)
            img2 = thresh_image_not[:, int(width/2):]
            img2_gray = cv2.bitwise_not(img2)

            contour1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            correct_contour1 = extract_correct_contour(contour1)
            rect1 = cv2.boundingRect(correct_contour1[0])
            contour2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            correct_contour2 = extract_correct_contour(contour2)
            rect2 = cv2.boundingRect(correct_contour2[0])

            digit1 = self.extract_digit_from_contour(rect1, img1_gray)
            digit2 = self.extract_digit_from_contour(rect2, img2_gray)

            total_value += digit1
            total_value += digit2

        else:

            (correct_contours, new_rects) = zip(*sorted(zip(correct_contours, rects),
                                                        key=lambda b: b[1][0], reverse=False))

            digit_image_gray = cv2.bitwise_not(thresh_image_not)

            for contour in correct_contours:

                rect = cv2.boundingRect(contour)
                # Draw the rectangles
                # cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

                digit = self.extract_digit_from_contour(rect, digit_image_gray)
                total_value += digit

        return total_value

    def extract_digit_from_contour(self, rect, img):

        if rect[1] < 2 or rect[0] < 2:
            roi = img[rect[1]:rect[1] + rect[3] + 3, rect[0]:rect[0] + rect[2] + 2]
        else:
            roi = img[rect[1] - 2:rect[1] + rect[3] + 3, rect[0] - 2:rect[0] + rect[2] + 2]

        _roi_h, _roi_w = roi.shape[:2]

        _a = max(_roi_h, _roi_w)
        _b = min(_roi_h, _roi_w)
        ones_img = np.ones((_a, _a), dtype=np.uint8) * 255
        if _roi_h > _roi_w:
            ones_img[:, (_a - _b) // 2: (_a - _b) // 2 + _b] = roi
        else:
            ones_img[(_a - _b) // 2: (_a - _b) // 2 + _b, :] = roi

        roi_gray_84 = cv2.resize(ones_img, (84, 84))

        roi_gray_dilate = cv2.dilate(roi_gray_84, kernel2, iterations=2)
        roi_gray_28 = cv2.resize(roi_gray_dilate, (28, 28), interpolation=cv2.INTER_CUBIC)
        roi_gray_not = cv2.bitwise_not(roi_gray_28)

        pred = self.model.predict(roi_gray_not.reshape(1, 28, 28, 1))
        value = str(pred.argmax())

        return value


def extract_correct_contour(cnts):

    correct_contours = []
    for i, contour in enumerate(cnts):

        area = cv2.contourArea(contour)
        if area > THRESH_AREA:
            correct_contours.append(contour)

    return correct_contours
