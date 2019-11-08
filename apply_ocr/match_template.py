import os
import cv2
import numpy as np

from google_vision import ExtractGoogleOCR

ROLL_LEFT = 0.17
ROLL_TOP = 0.12
ROLL_RIGHT = 0.33
ROLL_BOTTOM = 0.16

TOTAL_LEFT = 0.58
TOTAL_TOP = 0.63
TOTAL_RIGHT = 0.7
TOTAL_BOTTOM = 0.67

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


class MatchTemplate:

    def __init__(self, path):

        self.dirpath = path
        self.temp_path = os.path.join(path, "source/Template.png")
        self.crop_path = os.path.join(self.dirpath, "source/cropped.jpg")
        self.google_vision = ExtractGoogleOCR(self.dirpath)

    def match_template(self, image):

        MASK_THRESH = 165

        temp_img = cv2.imread(self.temp_path)
        temp_width = temp_img.shape[1]
        temp_height = temp_img.shape[0]

        img = image
        # img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        _, mask_img = cv2.threshold(img, thresh=MASK_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_img_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        boundingRect = cv2.boundingRect(sorted_contours[0])
        paper_img = image[boundingRect[1]:boundingRect[1] + boundingRect[3],
                        boundingRect[0]:boundingRect[0] + boundingRect[2]]
        cv2.imwrite(self.crop_path, paper_img)
        google_json = self.google_vision.get_json_google_from_jpg(self.crop_path)

        for i in range(len(google_json)):

            if google_json[i]['description'] == "No" and google_json[i + 1]['description'] == ":":
                roll_first_num = i + 1
            if google_json[i]['description'] == "Name":
                roll_last_num = i
            if google_json[i]['description'] == "Total":
                total_horizontal_bound = google_json[i]['boundingPoly']
                total_top = total_horizontal_bound['vertices'][0]['y'] - 1
                total_bottom = total_horizontal_bound['vertices'][2]['y'] + 11
            if google_json[i]['description'] == "Date":
                total_vertical_bound1 = google_json[i]['boundingPoly']
                total_right = total_vertical_bound1['vertices'][0]['x'] + 3
            if google_json[i]['description'] == "Equations":
                total_vertical_bound2 = google_json[i]['boundingPoly']
                total_left = total_vertical_bound2['vertices'][1]['x'] - 3

        roll_str = ""
        for i in range(roll_first_num + 1, roll_last_num):

            roll_str += google_json[i]['description']

        total_img = paper_img[total_top:total_bottom, total_left:total_right]

        return roll_str, total_img


def align_images(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h
