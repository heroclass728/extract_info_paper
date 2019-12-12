import json
import os
import base64
import requests

from datetime import datetime


class ExtractGoogleOCR:

    def __init__(self, path):
        """
            Constructor for the class
        """
        self.my_dir = path
        self.google_key = self.load_text(os.path.join(self.my_dir, 'source', 'vision_key.txt'))

    @staticmethod
    def load_json(filename):
        json_file = open(filename)
        json_data = json.load(json_file)
        return json_data

    @staticmethod
    def load_text(filename):
        if os.path.isfile(filename):
            file1 = open(filename, 'r')
            text = file1.read()
            file1.close()
        else:
            text = ''

        return text

    @staticmethod
    def save_text(filename, text):
        file1 = open(filename, 'w')
        file1.write(text.decode(encoding="utf-8"))
        file1.close()

    @staticmethod
    def rm_file(filename):
        if os.path.isfile(filename):
            os.remove(filename)

    @staticmethod
    def __make_request_json(img_file, output_filename, detection_type='text'):

        with open(img_file, 'rb') as image_file:
            content_json_obj = {'content': base64.b64encode(image_file.read()).decode('UTF-8')}
            # content_json_obj = {'content': base64.b64encode(image_file.read())}

        if detection_type == 'text':
            feature_json_arr = [{'type': 'TEXT_DETECTION'}, {'type': 'DOCUMENT_TEXT_DETECTION'}]
        elif detection_type == 'logo':
            feature_json_arr = [{'type': 'LOGO_DETECTION'}]
        else:
            feature_json_arr = [{'type': 'TEXT_DETECTION'}, {'type': 'DOCUMENT_TEXT_DETECTION'}]

        request_list = {'features': feature_json_arr, 'image': content_json_obj}

        # Write the object to a file, as json
        with open(output_filename, 'w') as output_json:
            json.dump({'requests': [request_list]}, output_json)

    def __get_text_info(self, json_file, detection_type='text'):

        data = open(json_file, 'rb').read()

        response = requests.post(
            url='https://vision.googleapis.com/v1/images:annotate?key=' + self.google_key,
            data=data,
            headers={'Content-Type': 'application/json'})

        ret_json = json.loads(response.text)
        ret_val = ret_json['responses'][0]

        if detection_type == 'text' and 'textAnnotations' in ret_val:
            return ret_val['textAnnotations']
        elif detection_type == 'logo' and 'logoAnnotations' in ret_val:
            return ret_val['logoAnnotations']
        else:
            return None

    def get_json_google_from_jpg(self, img_file, detection_type='text'):

        temp_json = "temp" + str(datetime.now().microsecond) + '.json'

        # --------------------- Image crop and rescaling, then ocr ------------------
        if img_file is None:
            ret_json = None
        else:
            self.__make_request_json(img_file, temp_json, detection_type)
            ret_json = self.__get_text_info(temp_json, detection_type)
            # ret_json = self.__get_google_request(temp_json, detection_type)

        # --------------------- delete temporary files -------------------------------
        self.rm_file(temp_json)
        # self.save_text(self.google_key, "")
        if ret_json is not None and detection_type == 'text':
            # for i in range(len(ret_json)):
            #     ret_json[i]['description'] = self.conv_str(ret_json[i]['description'])

            self.save_text('a_ocr.txt', ret_json[0]['description'].encode('utf-8'))

        return ret_json
