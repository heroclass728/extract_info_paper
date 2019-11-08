import os
import cv2
import numpy as np
import math

from apply_ocr.match_template import MatchTemplate
from apply_ocr.extract_handwritten_digit import DetectDigits
from keras.models import load_model
from google_vision import ExtractGoogleOCR


class DetectText:

    def __init__(self, path):

        self.dirpath = path

        self.total_path = os.path.join(self.dirpath, 'source/total2.jpg')

        # Load the classifier
        model_path = os.path.join(self.dirpath, 'model/digit_model.h5')
        self.model = load_model(model_path)

        self.matTemp = MatchTemplate(self.dirpath)
        self.detectDigit = DetectDigits(self.dirpath, self.model)

    def detect_text(self):

        CONTOUR_THRESH_VALUE = 80
        CONTOUR_THRESH_COUNT = 50

        video_path = os.path.join(self.dirpath, 'source/1.mp4')
        cap = cv2.VideoCapture(video_path)

        (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
        if (int(major_ver)) < 3:
            frame_rate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            frame_rate = cap.get(cv2.CAP_PROP_FPS)

        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=8)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        _, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        new_paper = False
        frame_cnt = 0
        frame_limit = int(3 * frame_rate)
        distance_flux = []
        while True:

            ret, new_frame = cap.read()
            if not ret:
                break

            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

            ret, fnd_contour_gray = cv2.threshold(new_gray, CONTOUR_THRESH_VALUE, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fnd_contour_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contour_len = len(contours)

            # calc distance
            diff_x, diff_y = 0, 0
            if p0 is not None and p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.circle(old_gray, (a, b), 3, (0, 0, 255), -1)
                    size = good_new.shape[0]
                    diff_x += (a - c) / size
                    diff_y += (b - d) / size

            distance = np.sqrt(diff_x ** 2 + diff_y ** 2)
            if len(distance_flux) <= 10:

                distance_flux.append(round(distance))
            else:
                distance_flux.pop(0)
                distance_flux.append(round(distance))

            ret_value = estimate_distances(distance_flux)

            if ret_value is True and contour_len > CONTOUR_THRESH_COUNT:

                new_paper = True
                frame_cnt += 1

            elif ret_value is False and new_paper is True:

                new_paper = False
                frame_cnt = 0

            if frame_cnt == frame_limit:

                avg_meaning_frame = new_frame
                roll, total_img = self.matTemp.match_template(avg_meaning_frame)
                total = self.detectDigit.detect_handwritten_digits(total_img)
                print("roll number:", roll)
                print("total marks:", total)

            old_gray = new_gray

        cap.release()
        f_path = os.path.join(self.dirpath, "source", "vision_key.txt")
        ExtractGoogleOCR(self.dirpath).save_text(f_path, "")


def calculate_distance_between_point(pts1, pts2):
    total_dst = 0
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        x1, y1 = pt1.ravel()
        x2, y2 = pt2.ravel()
        dst = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_dst += dst

    total_dst = total_dst / len(pts1)

    return total_dst


def estimate_distances(flux):

    for dist in flux:

        if dist > 5:
            ret_val = False
            break
        ret_val = True

    return ret_val
