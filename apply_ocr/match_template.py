import os
import cv2
import numpy as np
import pytesseract

from apply_ocr.google_vision import ExtractGoogleOCR

ROLL_LEFT = 0.17
ROLL_TOP = 0.12
ROLL_RIGHT = 0.33
ROLL_BOTTOM = 0.16

TOTAL_LEFT = 0.25
TOTAL_RIGHT = 0.75
TOTAL_TOP = 0.5
TOTAL_BOTTOM = 0.9

TEMPLATE_WIDTH = 661
TEMPLATE_HEIGHT = 929

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


class MatchTemplate:

    def __init__(self, path):

        self.dirpath = path
        self.paper_path = os.path.join(self.dirpath, "source", "paper.jpg")
        self.crop_path = os.path.join(self.dirpath, "source", "cropped.jpg")
        self.first_crop_path = os.path.join(self.dirpath, "source", "first_cropped.jpg")
        self.table_path = os.path.join(self.dirpath, "source", "table.jpg")
        self.google_vision = ExtractGoogleOCR(self.dirpath)

    def match_template(self, image):

        MASK_THRESH = 100

        img = image
        _, mask_img = cv2.threshold(img, thresh=MASK_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        cv2.imwrite(self.paper_path, mask_img)
        mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_img_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        boundingRect = cv2.boundingRect(sorted_contours[0])

        bounding_points = [[boundingRect[0], boundingRect[1]], [boundingRect[0], boundingRect[1] + boundingRect[3]],
                           [boundingRect[0] + boundingRect[2], boundingRect[1]],
                           [boundingRect[0] + boundingRect[2], boundingRect[1] + boundingRect[3]]]

        corner_points = np.float32(extract_paper_corner(bounding_points, sorted_contours[0]))
        bounding_points = np.float32(bounding_points)
        M = cv2.getPerspectiveTransform(corner_points, bounding_points)
        img_dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

        paper_img = img_dst[boundingRect[1]:boundingRect[1] + boundingRect[3],
                    boundingRect[0]:boundingRect[0] + boundingRect[2]]
        cv2.imwrite(self.crop_path, paper_img)
        paper_img_width = paper_img.shape[1]
        paper_img_height = paper_img.shape[0]

        first_part_paper_img = paper_img[:int(paper_img_width * TOTAL_TOP), :]
        cv2.imwrite(self.first_crop_path, first_part_paper_img)
        first_google_json = self.google_vision.get_json_google_from_jpg(self.first_crop_path)

        table_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        table_thresh = cv2.adaptiveThreshold(table_image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 5)
        table_thresh_not = cv2.bitwise_not(table_thresh)
        table_contours, _ = cv2.findContours(table_thresh_not, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        table_sorted_contours = sorted(table_contours, key=cv2.contourArea, reverse=True)
        table_boundingRect = cv2.boundingRect(table_sorted_contours[0])
        # cv2.rectangle(image, (boundingRect[0], boundingRect[1]), (boundingRect[0] + boundingRect[2],
        #                                                           boundingRect[1] + boundingRect[3]), (0, 0, 255), 3)
        total_left = table_boundingRect[0] + int(table_boundingRect[2] * 2 / 3) + 3
        total_top = table_boundingRect[1] + int(table_boundingRect[3] * 5 / 6) + 1
        total_right = table_boundingRect[0] + table_boundingRect[2] - 3
        total_bottom = table_boundingRect[1] + table_boundingRect[3] - 1
        total_img_init = image[total_top:total_bottom, total_left:total_right]
        total_img = extract_non_grid_image(total_img_init)
        # table_img = paper_img[int(paper_img_width * TOTAL_TOP):int(paper_img_width * TOTAL_BOTTOM),
        #             int(paper_img_width * TOTAL_LEFT):int(paper_img_width * TOTAL_RIGHT)]
        # non_grid_img = extract_non_grid_image(table_img)
        # cv2.imwrite(self.table_path, non_grid_img)
        # # table_google_json = self.google_vision.get_json_google_from_jpg(self.table_path)
        # total_marks = extract_total_marks(non_grid_img)

        roll_first_num = 0
        roll_last_num = 0
        roll_str = ""
        roll_str_init = ""
        exam_type = ""
        try:
            for i in range(len(first_google_json)):

                if first_google_json[i]['description'] == "No" and first_google_json[i + 1]['description'] == ":":
                    roll_first_num = i + 1
                elif first_google_json[i]['description'] == "Name":
                    roll_last_num = i
                elif first_google_json[i]['description'] == "Equations":
                    exam_type = first_google_json[i + 1]['description']

            for i in range(roll_first_num + 1, roll_last_num):
                roll_str_init += first_google_json[i]['description']

            if "o" in roll_str_init:
                no_index = roll_str_init.rindex("o")
                roll_str = roll_str_init.replace(roll_str_init[:no_index + 1], "")
            else:
                roll_str = roll_str_init

        except Exception as e:

            print(e)

        return roll_str, total_img, exam_type


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


def extract_paper_corner(points, contour):
    corners = []
    for point in points:

        point_x = point[0]
        point_y = point[1]
        point_dist = []

        for cnt_pt in contour:
            cnt_pt_x = cnt_pt[0][0]
            cnt_pt_y = cnt_pt[0][1]

            dist = (point_x - cnt_pt_x) ** 2 + (point_y - cnt_pt_y) ** 2
            point_dist.append(dist)

        min_dist_index = np.argmin(point_dist)
        minim_pt = contour[min_dist_index]
        corners.append([minim_pt[0][0], minim_pt[0][1]])

    return corners


def extract_non_grid_image(image):

    # result_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    thresh_not = cv2.bitwise_not(thresh)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    remove_horizontal = cv2.morphologyEx(thresh_not, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), 1)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    remove_vertical = cv2.morphologyEx(thresh_not, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), 3)

    return thresh


def extract_total_marks(image):
    image_text = pytesseract.image_to_string(image)
    index_max = image_text.index("50")
    total = image_text[index_max + 2:]
    total.replace(" ", "")

    return total
