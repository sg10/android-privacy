import argparse
import base64
import json
import keras as K
import logging
import os
import pprint
import random
from datetime import datetime

from androguard.core.androconf import show_logging
from androguard.core.bytecodes.apk import APK
from io import BytesIO
from PIL import Image

from verifier import config
from verifier.neural.text2permission.datagen import get_predict_fn as get_predict_fn_t2p
from verifier.neural.text2permission.model import model_multiconv_1d, warnings
from verifier.preprocessing.permissionparser import PermissionParser
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.test.report_saver import ReportSaver
from verifier.util.explain.lime import detect_relevant_word_inputs
from verifier.util.strings import remove_html


class APKRunner:
    def __init__(self, report_folder, apk=None, txt=None, package_name=None):
        self.report_folder = report_folder
        self.apk_file = apk
        self.text_file = txt
        self.n_nearest_apps = 15
        self.n_top_terms = 25
        self.load_from_preprocessed = True  # only checks package name, not version, hash, etc.

        self.app_name = "Unknown App | " + datetime.today().strftime('%Y-%m-%d %H:%M')
        self.text = None
        self.package_name = package_name
        self.result_file_name = None
        self.report_saver = ReportSaver(report_folder=report_folder)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        show_logging(logging.ERROR)  # androguard
        warnings.simplefilter(action='ignore', category=FutureWarning)

    def run(self):

        if self.apk_file:
            self.get_apk_info()
        else:
            self.get_fallback_app_info()

        if self.text_file:
            self.read_text()
        else:
            self.get_fallback_description()

        print("-" * 80)
        print(self.app_name or self.apk_file)
        print("-" * 80)

        self.process_t2p()
        self.report_saver.save()

    def get_apk_info(self):
        apk = APK(self.apk_file)
        app_icon_file = apk.get_app_icon()
        app_icon_data = apk.get_file(app_icon_file)

        size = (256, 256)

        buffered = BytesIO()
        im = Image.open(BytesIO(app_icon_data))
        im = im.resize(size, Image.ANTIALIAS)
        im.save(buffered, "PNG")

        app_icon_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        self.package_name = apk.get_package()
        self.app_name = apk.get_app_name()

        self.report_saver.package_name = self.package_name
        self.report_saver.app_name = self.app_name
        self.report_saver.version = apk.get_androidversion_code()
        self.report_saver.app_icon = app_icon_b64

        permission_parser = PermissionParser(mode='groups')
        permission_values = permission_parser.transform(apk.get_permissions()).flatten().tolist()
        permission_labels = permission_parser.labels()
        self.report_saver.permissions_actual = {permission_labels[i]: bool(v) for i, v in enumerate(permission_values)}

    def get_fallback_app_info(self):
        db = SamplesDatabase.get()
        if self.package_name is None:
            raise RuntimeError("no package name!")
        self.app_name = db.read(self.package_name, 'title')
        self.report_saver.package_name = self.package_name
        self.report_saver.app_name = self.app_name
        self.report_saver.version = ""
        self.report_saver.app_icon = None
        self.report_saver.permissions_actual = db.read(self.package_name, 'permissions') or {}

    def get_fallback_description(self):
        self.text = SamplesDatabase.get().read(self.package_name, 'description_raw')

    def read_text(self):
        text = open(self.text_file, "r", encoding="utf-8").read()
        try:
            json_data = json.loads(text)
            text = remove_html(json_data['description_html'])
        except ValueError:
            pass

        self.text = text

    def process_t2p(self):
        self.report_saver.t2p = {}
        permission_parser = PermissionParser('groups')
        ml_model = model_multiconv_1d(permission_parser.count())

        ml_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        ml_model.load_weights(config.TrainedModels.text2permission)

        tokens, tokens_heat, predictions = detect_relevant_word_inputs(self.text,
                                                                       get_predict_fn_t2p(ml_model),
                                                                       permission_parser.labels())

        self.report_saver.t2p['tokens'] = tokens
        self.report_saver.t2p['tokens_heat'] = tokens_heat
        self.report_saver.t2p['permissions_pred'] = predictions

        K.backend.clear_session()


def is_a_dir(s):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)


def process_apk(file_txt, file_apk):
    parser = argparse.ArgumentParser(description='Android App Description Privacy Awareness')

    parser.add_argument('--apk',
                        help='Android APK file')
    parser.add_argument('--txt',
                        help='Description text file')

    parser.add_argument('--out',
                        help='folder for generated report')

    args = parser.parse_args(["--txt", file_txt,
                              "--apk", file_apk,
                              "--out", "/data/reports"])

    if args.txt is None or not os.path.isfile(args.txt):
        print("File does not exist: ", args.txt or "<TXT>")
        parser.print_help()
        return
    if args.apk is None or not os.path.isfile(args.apk):
        print("File does not exist: ", args.apk or "<APK>")
        parser.print_help()
        return

    if not args.txt or not args.apk:
        print("Need at least either APK file or description text file.")
        parser.print_help()
        return

    runner = APKRunner(args.out, apk=args.apk, txt=args.txt)
    runner.run()


def from_command_line():
    files = list(os.listdir("/data/test_apps/"))
    try:
        files_exist = list(os.listdir("/data/reports/apps/"))
    except:
        files_exist = []
    random.shuffle(files)
    # files = ['nu.mine.tmyymmt.aflashlight-97.apk']
    # test_packs = SamplesDatabase.get().filter(('set', '==', 'test'))
    # pprint.pprint(test_packs)
    # pprint.pprint(files)
    # asd
    for filename_apk in files:

        file_apk = os.path.join("/data/test_apps/", filename_apk)
        filename_json = filename_apk.rsplit("-", maxsplit=1)[0] + ".json"
        file_txt = os.path.join("/data/samples/metadata/", filename_json)

        # if filename_json in files_exist:
        #    print("exist: ", filename_json)
        #    continue

        try:
            process_apk(file_txt, file_apk)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    from_command_line()
