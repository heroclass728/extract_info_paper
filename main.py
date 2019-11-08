import os

from apply_ocr.detect_text import DetectText


if __name__ == '__main__':

    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    DetectText(_cur_dir).detect_text()
