import keras
from keras.layers import *

from sklearn.metrics import classification_report, precision_recall_fscore_support

from verifier import config
from verifier.preprocessing.permissionparser import PermissionParser
from verifier.preprocessing.pretrained_embedding import PreTrainedEmbeddings
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.neural.text2permission.datagen import get_predict_fn
from verifier.util.metrics import class_report_fbeta
from verifier.util.explain.cnn_heatmap import HeatmapCalculator
from verifier.util.explain.lime import detect_relevant_word_inputs
from verifier.util.string_table_builder import StringTable


class PrintSamples(keras.callbacks.Callback):

    def __init__(self, generator, print_per_label=2):
        super().__init__()
        self.generator = generator
        self.permission_parser = PermissionParser('groups')
        num_batches = min(config.Text2PermissionClassifier.batch_size//self.permission_parser.count(),
                          len(self.generator))
        self.batch_indices = np.random.choice(len(self.generator), num_batches, replace=False)
        self.threshold_true = 0.65
        self.print_per_label = print_per_label
        self.db = SamplesDatabase.get()

    def on_epoch_end(self, epoch, logs=None):
        self.predict_batch()

    def predict_batch(self):
        num_permissions = self.permission_parser.count()

        # minimum number of positive (=1) samples per label (permission) to show

        num_print_left = {i: self.print_per_label for i in range(num_permissions)}

        output_table = StringTable()
        labels = self.permission_parser.labels()
        output_table.set_headline(labels)

        for batch_idx in self.batch_indices:
            X, y, packages = self.generator.get_item_and_package(batch_idx)
            p = self.model.predict(X)

            for i, package in enumerate(packages):
                predicted = np.rint(p[i])
                real = np.rint(y[i])

                print_this_sample = False

                for j in range(num_permissions):
                    if real[j] == 1 and num_print_left[j] > 0:
                        num_print_left[j] -= 1
                        print_this_sample = True

                if print_this_sample:
                    evals = [" %.2f--%d %s" %
                             (p[i][k], real[k], "ok" if predicted[k] == real[k] == 1 else "")
                             for k in range(real.shape[0])]
                    output_table.add_cells(evals)

                    output_table.add_cell(package)
                    output_table.add_cell(self.db.read(package, 'title'))

                    raw_text = SamplesDatabase.get().read(package, 'description_raw')
                    output_table.add_cell(self.get_top_words_per_class(raw_text))

                    output_table.new_row()

        output_table.set_cell_length(-1, 1000)
        for row in output_table.create_table(return_rows=True):
            print(row)

    def get_top_words_per_class(self, text_raw):
        #split_exp = PreTrainedEmbeddings.get().get_delimiter_regex_pattern()
        tokens, tokens_heat, preds = detect_relevant_word_inputs(text_raw.lower(),
                                                                 get_predict_fn(self.model),
                                                                 PermissionParser(mode='groups').labels())
        full_str = ""
        for class_name, heats in tokens_heat.items():
            tokens_with_heats = ["%s=%d%%" % (tokens[token_idx], heat) for token_idx, heat in heats if heat > 50][:5]
            if len(tokens_with_heats) > 0:
                full_str += "[%s] %s  " % (class_name[:5], " ".join(tokens_with_heats))

        return full_str


class PrintPerClassMetrics(keras.callbacks.Callback):
    ''' outputs the performance of the validation set for each target flass '''

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.permission_parser = PermissionParser(mode='groups')

    def on_epoch_end(self, epoch, logs=None):
        print("##  VALIDATION METRICS")
        self.predict_batch()

    def predict_batch(self, print_report=True):
        num_samples_total = len(self.generator)*self.generator.batch_size
        num_permissions = self.permission_parser.count()
        y_true = np.zeros(shape=(num_samples_total, num_permissions))
        y_pred = np.zeros(shape=(num_samples_total, num_permissions))

        i = 0
        for X_batch, y_true_batch in self.generator:
            y_pred_batch = self.model.predict(X_batch)
            y_pred_batch = np.rint(y_pred_batch)

            n_batch_samples = y_pred_batch.shape[0]  # batch size or less (ultimate batch)

            y_true[i:i+n_batch_samples, :] = y_true_batch[:]
            y_pred[i:i+n_batch_samples, :] = y_pred_batch[:]

            i += self.generator.batch_size

        if y_true.shape[0] > 0 and y_pred.shape[0] > 0:
            report = class_report_fbeta(y_pred, y_true, self.permission_parser.labels(), 0.5, print_output=print_report)

            return report

        return None

