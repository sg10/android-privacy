import json
import random
import os
import keras as K

from verifier import config
from verifier.neural.text2permission.datagen import get_predict_fn as get_predict_fn_t2p
from verifier.neural.text2permission.model import model_multiconv_1d, warnings
from verifier.preprocessing.permissionparser import PermissionParser
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.test.report_saver import ReportSaver
from verifier.util.explain.lime import detect_relevant_word_inputs
from verifier.util.strings import remove_html


class TestsetEvaluator:

    def __init__(self, package_names, report_folder):
        self.n_top_terms = 25

        self.package_names = package_names
        self.report_folder = report_folder
        self.result_file_name = None
        self.report_saver = ReportSaver(report_folder=report_folder)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.simplefilter(action='ignore', category=FutureWarning)

    def run(self):
        self.process_t2p()

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


def from_samples_db():
    test_packages = SamplesDatabase.get().filter(('set', '==', 'test'))

    # for p in test_packages:
    #     print("-" * 50)
    #     print(" - ", SamplesDatabase.get().read(p, 'title'))
    #     print("   ", p)
    #     print()
    #     print(SamplesDatabase.get().read(p, 'description_raw'))
    #     print()

    test_packages_1 = ["com.dl.photo.loveframes",
                     "mp3.tube.pro.free",
                     "de.stohelit.folderplayer",
                     "com.avg.cleaner",
                     "com.gau.go.launcherex.gowidget.gopowermaster",
                     "ru.mail",
                     "de.shapeservices.impluslite",
                     "kik.android",
                     "com.hrs.b2c.android",
                     "com.nqmobile.antivirus20.multilang",
                     "com.yellowbook.android2",
                     "com.antivirus.tablet",
                     "taxi.android.client",
                     "com.qihoo.security",
                     "com.jb.gokeyboard.plugin.emoji",
                     "com.niksoftware.snapseed",
                     "com.forshared.music",
                     "mobi.infolife.eraser",
                     "com.hulu.plus",
                     "com.vevo",
                     "com.mobisystems.office",
                     "com.whatsapp",
                     "com.dropbox.android",
                     "com.yahoo.mobile.client.android.yahoo",
                     "com.jessdev.hdcameras",
                     "com.slacker.radio",
                     "com.jb.mms.theme.springtime",
                     "ru.zdevs.zarchiver",
                     "com.newsoftwares.folderlock_v1"]

    test_packages = list(set(test_packages).difference(test_packages_1))

    random.shuffle(test_packages)

    runner = TestsetEvaluator(report_folder="data/reports", package_names=test_packages)
    runner.run()


if __name__ == "__main__":
    from_samples_db()
