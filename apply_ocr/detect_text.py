import os
import cv2
import numpy as np
import math
import time
import threading

from utils.constants import CUR_DIR, FRAME_RATE
from apply_ocr.match_template import MatchTemplate
from apply_ocr.extract_handwritten_digit import DetectDigits
from keras.models import load_model
from manage_database.write_data import ManageDatabase
from manage_database.connect_db import connect_db


class DetectText(threading.Thread):

    def __init__(self, queue):
        super(DetectText, self).__init__()

        self.dirpath = CUR_DIR

        self._stopped = threading.Event()
        self._stopped.clear()
        self._paused = threading.Event()
        self._paused.clear()

        self.roll = ""
        self.total = ""
        self.alert = ""
        self.exam_type = ""
        # self.total_path = os.path.join(self.dirpath, 'source/total2.jpg')

        # Load the classifier
        model_path = os.path.join(self.dirpath, 'model/digit_model.h5')
        self.model = load_model(model_path)

        self.matTemp = MatchTemplate(self.dirpath)
        self.detectDigit = DetectDigits(self.dirpath, self.model)
        self.db = connect_db(self.dirpath)
        self.manage_database = ManageDatabase(self.db)

        self.frame_queue = queue

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def stop(self):
        self._stopped.set()

    def run(self):

        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=8)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        old_frame = None
        while True:
            try:
                old_frame = self.frame_queue.get()
            except self.frame_queue.Empty:
                time.sleep(1)
            if old_frame is not None:
                break

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        new_paper = False
        frame_cnt = 0
        frame_limit = int(3 * FRAME_RATE)
        distance_flux = []

        while True:

            # ret, new_frame = cap.read()
            if self._stopped.isSet():
                break
            elif self._paused.isSet():
                time.sleep(.1)
                continue

            try:
                new_frame = self.frame_queue.get()
                if self._stopped.isSet():
                    break
            except self.frame_queue.Empty:
                time.sleep(0.2)
                continue

            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

            _, new_gray_thresh = cv2.threshold(new_gray, 200, 255, cv2.THRESH_BINARY)

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

            if ret_value is False:

                new_paper = True

            elif ret_value is True and new_paper is True:

                frame_cnt += 1

            if frame_cnt == frame_limit:
                avg_meaning_frame = new_frame
                self.roll, total_img, self.exam_type = self.matTemp.match_template(avg_meaning_frame)
                self.total = self.detectDigit.detect_handwritten_digits(total_img)
                print(self.roll, self.exam_type, self.total)
                if self.roll == "" or self.total == "" or self.exam_type == "":

                    self.alert = "Not Saved into Database"

                else:
                    # print("roll number:", self.roll)
                    # # cv2.imwrite(self.total_path, total_img)
                    # self.total = self.detectDigit.detect_handwritten_digits(total_img)
                    #
                    # print("total marks:", self.total)
                    self.manage_database.insert_data(self.roll, self.total, self.exam_type)
                    self.alert = "Successfully Saved into Database"

                new_paper = False
                frame_cnt = 0

            old_gray = new_gray

        self.db.close()


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
    ret_val = False
    for dist in flux:

        if dist > 1:
            ret_val = False
            break
        ret_val = True

    return ret_val
