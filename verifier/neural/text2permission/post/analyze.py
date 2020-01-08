import json
import pprint

import keras
import os
import warnings
import numpy as np

from verifier import config
from verifier.neural.text2permission.callbacks import PrintPerClassMetrics
from verifier.neural.text2permission.datagen import get_predict_fn, Generator
from verifier.neural.text2permission.model import model_multiconv_1d
from verifier.preprocessing.permissionparser import PermissionParser
from verifier.preprocessing.pretrained_embedding import PreTrainedEmbeddings
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.util import metrics
from verifier.util.explain.lime import detect_relevant_word_inputs
from verifier.util.train import get_t2p_word_embedding_type


class DescriptionPermissionAnalysis:

    def __init__(self):
        self.descriptions_limed = None
        self.db = SamplesDatabase.get()

    def load_limed(self):
        self.descriptions_limed = json.load(open(config.Text2PermissionClassifier.test_set_lime, "r"))

    def generate(self):
        warnings.simplefilter('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        if get_t2p_word_embedding_type() == "word2vec":
            key_description_num_tokens = "description_num_tokens_word2vec"
        else:
            key_description_num_tokens = "description_num_tokens_glove"
        package_names = self.db.filter(('lang', '==', 'en'),
                                  ('set', '==', 'test'),
                                  (key_description_num_tokens, '>=', 20))

        permission_parser = PermissionParser(mode='groups')
        model = model_multiconv_1d(permission_parser.count())

        model.compile(loss="binary_crossentropy",
                      optimizer='Adam',
                      metrics=[metrics.fb_micro, metrics.fb_micro, metrics.precision, metrics.recall])

        model.summary()

        # keras.utils.plot_model(model, '/home/me/model.png')

        model.load_weights(config.TrainedModels.text2permission)

        descriptions_limed = {}

        for i, package in enumerate(package_names):
            text_raw = self.db.read(package, 'description_raw')

            tokens, tokens_heat, preds = detect_relevant_word_inputs(text_raw,
                                                                     get_predict_fn(model),
                                                                     permission_parser.labels())
            descriptions_limed[package] = {
                'tokens': tokens,
                'tokens_heat': tokens_heat,
                'preds': preds
            }

            if i % 30 == 0:
                print("%d%%" % (i/len(package_names)*100))

            # if i == 3: break

        json.dump(descriptions_limed, open(config.Text2PermissionClassifier.test_set_lime, 'w'))

        self.descriptions_limed = descriptions_limed

        test_generator = Generator(packages=package_names, batch_size=64)
        print_metrics = PrintPerClassMetrics(test_generator)
        print_metrics.model = model
        print_metrics.predict_batch()

    def list_apps(self):
        print("*" * 10, "  PACKAGES   ", "*" * 10)
        print()
        for package_name in self.descriptions_limed.keys():
            print("%40s      %s" % (package_name, self.db.read(package_name, 'title')))

    def top_words_per_permission_group(self):
        print("*" * 10, "  TOP WORDS  ", "*" * 10)
        print()

        permission_groups = list(self.descriptions_limed.values())[0]['preds'].keys()
        #permission_groups = ['call_log']

        top_words = {}

        for permission_group in permission_groups:
            for package_name, lime_data in self.descriptions_limed.items():

                if lime_data['preds'][permission_group] > 0.7:

                    for token_id, value in lime_data['tokens_heat'][permission_group]:
                        if value < 50: continue

                        word = lime_data['tokens'][token_id].lower()
                        top_words[permission_group] = top_words.get(permission_group, {})
                        top_words[permission_group][word] = top_words[permission_group].get(word, [])

                        top_words[permission_group][word] += [value]

            lst = list(top_words[permission_group].items())
            lst.sort(key=lambda w: np.mean(w[1]), reverse=True)
            lst = list(filter(lambda w: len(w[1]) > 4, lst))
            top_words[permission_group] = lst

        for permission_group in permission_groups:
            print("------- ", permission_group)
            for i, (word, vals) in enumerate(top_words[permission_group][:20]):
                print("%3d   %.2f   %s" % (i, float(np.mean(vals)), word))

    def latex_text_description(self, package_name, permission_group, color='gray'):
        tokens = self.descriptions_limed[package_name]['tokens']
        heats = self.descriptions_limed[package_name]['tokens_heat'][permission_group]

        out = ""

        first_occurrence_token = {}
        first_occurrence_idx = {}
        for idx, token in enumerate(tokens):
            if token not in first_occurrence_token:
                first_occurrence_token[token] = idx
            first_occurrence_idx[idx] = first_occurrence_token[token]

        for idx, token in enumerate(tokens):
            heat = list(filter(lambda x: x[0] == idx or x[0] == first_occurrence_idx[idx], heats))
            if len(heat) == 0 or heat[0][1] < 20:
                out += token
            else:
                intensity = round(heat[0][1])
                out += "\colorbox{%s!%d}{%s}" % (color, intensity, token)

        print(out)


def run():
    dpa = DescriptionPermissionAnalysis()
    #dpa.generate()
    dpa.load_limed()
    #dpa.list_apps()
    #dpa.top_words_per_permission_group()

    package = "de.lieferheld.android"
    permissions = ["location", "camera"]

    for p in permissions:
        print("-" * 50)
        print(p, "       ", package)
        dpa.latex_text_description(package, p)

    print("finished.")


if __name__ == "__main__":
    run()
