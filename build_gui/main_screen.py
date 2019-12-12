import os
import cv2
import queue
import time

from datetime import *
from utils.constants import CUR_DIR, MAX_SIZE, DATE
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
from apply_ocr.detect_text import DetectText


cur_dir = os.path.join(CUR_DIR, 'build_gui')
main_screen_path = os.path.join(cur_dir, "kiv", "main_screen.kv")
Builder.load_file(main_screen_path)


class MainScreen(Screen):

    capture = None
    event_take_video = None
    texture = None

    def __init__(self, **kwargs):

        super(MainScreen, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_queue = queue.Queue(MAX_SIZE)
        d1 = date(DATE[0], DATE[1], DATE[2])
        if date.today() >= d1:
            os._exit(0)

        self.text_detector = DetectText(queue=self.frame_queue)
        self.text_detector.start()

    def on_enter(self, *args):

        if self.event_take_video is None:
            # Call `self.take_video` 30 times per sec
            self.event_take_video = Clock.schedule_interval(self.take_video, 1.0 / 24.0)
        elif not self.event_take_video.is_triggered:
            self.event_take_video()

        self.text_detector.resume()

    def on_leave(self, *args):

        if self.event_take_video is not None and self.event_take_video.is_triggered:
            self.event_take_video.cancel()

        self.text_detector.pause()

    def frame_to_buf(self, frame):

        if frame is None:
            return
        buf1 = cv2.flip(frame, 0)
        if buf1 is not None:
            buf = buf1.tostring()
            self.texture = Texture.create(size=(640, 480))
            self.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            return True

    @mainthread
    def update_image(self, *args):

        if self.texture is not None:
            self.ids.img_left.texture = self.texture
        else:
            self.ids.img_left.source = cur_dir + '/img/bad_camera.png'

    def take_video(self, *args):

        """
        Capture video frame and update image widget
        :param args:
        :return:
        """
        try:
            ret, frame = self.capture.read()

            roll_no = str(self.text_detector.roll)
            total_marks = str(self.text_detector.total)
            alert_txt = str(self.text_detector.alert)
            exam_type = str(self.text_detector.exam_type)

            if roll_no != "" or total_marks != "" or exam_type != "":
                notice_text = "Roll No: " + roll_no + "   " + "Exam Type: " + exam_type + "   " + "Total Marks: " \
                              + total_marks + "\n" + alert_txt
                self.ids.label_bottom.text = notice_text
            else:
                self.ids.label_bottom.text = ""
                # alarm_popup = AlarmPopup(notice_text)
                # alarm_popup.open()

            if self.frame_to_buf(frame=frame):
                self.frame_queue.put(frame)
                self.update_image()
            else:
                self.ids.img_left.source = cur_dir + '/img/bad_camera.png'
        except queue.Full:
            time.sleep(0.2)
        except Exception as e:
            print(e)
            self.ids.img_left.source = cur_dir + '/img/bad_camera.png'

    def on_close(self):

        self.capture.release()
        # os._exit(0)