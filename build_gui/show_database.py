import os

from utils.constants import CUR_DIR
from manage_database.connect_db import connect_db
from manage_database.write_data import ManageDatabase

from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.button import Button
from kivy.properties import BooleanProperty, ListProperty, StringProperty, ObjectProperty
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.popup import Popup

cur_dir = os.path.join(CUR_DIR, 'build_gui')
show_database_path = os.path.join(cur_dir, "kiv", "show_database.kv")
Builder.load_file(show_database_path)


class WarningPopup(Popup):

    label = StringProperty()

    def __init__(self, label, **kwargs):
        super(WarningPopup, self).__init__(**kwargs)
        self.set_description(label)

    def set_description(self, label):
        self.label = label


class DataRemovePopup(Popup):

    obj = ObjectProperty(None)

    def __init__(self, obj, **kwargs):
        super(DataRemovePopup, self).__init__(**kwargs)
        self.obj = obj


class DataAddPopup(Popup):

    obj = ObjectProperty(None)

    def __init__(self, obj, **kwargs):
        super(DataAddPopup, self).__init__(**kwargs)
        self.obj = obj


class DataUpdatePopup(Popup):

    obj = ObjectProperty(None)
    obj_text = StringProperty("")

    def __init__(self, obj, **kwargs):

        super(DataUpdatePopup, self).__init__(**kwargs)
        self.obj = obj
        self.obj_text = obj.text


class SelectableRecycleGridLayout(FocusBehavior, LayoutSelectionBehavior, RecycleGridLayout):
    """ Adds selection and focus behaviour to the view. """


class SelectableButton(RecycleDataViewBehavior, Button):
    """ Add selection support to the Button """

    index = None
    data = []
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(SelectableButton, self).__init__(**kwargs)
        self.db = connect_db(CUR_DIR)
        self.manage_database = ManageDatabase(self.db)

    def refresh_view_attrs(self, rv, index, data):
        """ Catch and handle the view changes """

        self.index = index
        self.data = data

        return super(SelectableButton, self).refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        """ Add selection on touch down """

        if super(SelectableButton, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        """ Respond to the selection of items in the view. """

        self.selected = is_selected

    def on_press(self):

        popup = DataUpdatePopup(self)
        popup.open()

    def update_changes(self, txt):

        self.text = txt
        field_index = self.index % 4
        id_index = self.index // 4
        field = ""
        if field_index == 1:
            field = "roll_number"
        elif field_index == 2:
            field = "exam_type"
        elif field_index == 3:
            field = "total_marks"

        self.manage_database.update_data(field, str(txt), id_index)


class ShowDatabase(Screen):

    data_items = ListProperty([])
    db = ""
    manage_database = ManageDatabase

    def on_enter(self, *args):

        self.db = connect_db(CUR_DIR)
        self.manage_database = ManageDatabase(self.db)
        rows = self.manage_database.read_data()

        # create data_items
        for row in rows:
            for col in row:
                self.data_items.append(col)

    def on_leave(self, *args):

        self.data_items.clear()
        self.db.close()

    def add_data(self):

        popup = DataAddPopup(self)
        popup.open()

    def add_changes(self, roll_no, total_marks, exam_type):

        if roll_no != "" and total_marks != "" and exam_type != "":

            user_id = int(self.data_items[-4]) + 1
            self.data_items.append(user_id)
            self.data_items.append(roll_no)
            self.data_items.append(exam_type)
            self.data_items.append(total_marks)
            self.manage_database.insert_data(roll_no, total_marks, exam_type)

        else:

            warning_popup = WarningPopup("Please insert all fields")
            warning_popup.open()

    def remove_data(self):

        remove_popup = DataRemovePopup(self)
        remove_popup.open()

    def remove_changes(self, user_id):

        try:
            if user_id != "":

                user_id_indices = [n for n, item in enumerate(self.data_items) if item == int(user_id)]
                user_id_index = ""
                for index in user_id_indices:
                    if index % 4 == 0:
                        user_id_index = index

                for j in range(4):
                    self.data_items.pop(user_id_index)

                self.manage_database.delete_data(user_id)

            else:

                warning_popup = WarningPopup("Please insert USER_ID fields")
                warning_popup.open()

        except Exception as e:
            print(e)
