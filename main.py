import os

from apply_ocr.detect_text import DetectText
from utils.constants import MAIN_SCREEN, SHOW_DATABASE
from build_gui.main_screen import MainScreen
from build_gui.show_database import ShowDatabase
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager


class DetectPaperTool(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.main_screen = MainScreen(name=MAIN_SCREEN)
        self.show_database = ShowDatabase(name=SHOW_DATABASE)

        screens = [

            self.main_screen,
            self.show_database,
        ]

        self.sm = ScreenManager()
        for screen in screens:
            self.sm.add_widget(screen)

    def build(self):
        self.sm.current = MAIN_SCREEN

        return self.sm

    def on_stop(self):
        self.main_screen.text_detector.stop()
        self.main_screen.text_detector.join()


if __name__ == '__main__':

    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    # DetectText().run()
    DetectPaperTool().run()
