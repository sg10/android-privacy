# has been removed from keras in v2,
# taken from https://github.com/keras-team/keras/issues/5400

from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.contrib.framework import argsort

from verifier.util.string_table_builder import StringTable


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def get_fbeta_micro(beta=1.0):

    def fbeta_micro(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)

        return (1+beta*beta) * (p * r) / ((beta*beta*p) + r + K.epsilon())

    return fbeta_micro


def get_fbeta_macro(beta):

    def fbeta_macro(y_true, y_pred):
        # https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
        y_pred = K.round(y_pred)

        true_positives = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        false_positives = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        false_negatives = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = true_positives / (true_positives + false_positives + K.epsilon())
        r = true_positives / (true_positives + false_negatives + K.epsilon())

        f1 = (1+beta*beta) * p * r / ((beta*beta*p) + r + K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

        return K.mean(f1)

    return fbeta_macro


def fb_micro(y_true, y_pred):
    return get_fbeta_micro(0.5)(y_true, y_pred)


def fb_macro(y_true, y_pred):
    return get_fbeta_macro(0.5)(y_true, y_pred)


def class_report_fbeta(y_pred, y_true, labels, beta, print_output=False):
    precision, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred, beta=beta)
    macro_precision, macro_recall, macro_fbeta, macro_support = \
        precision_recall_fscore_support(y_true, y_pred, beta=beta, average='macro')
    micro_precision, micro_recall, micro_fbeta, micro_support = \
        precision_recall_fscore_support(y_true, y_pred, beta=beta, average='micro')

    report = {}

    for i, label in enumerate(labels):
        report[label] = {
            'precision': precision.tolist()[i],
            'recall': recall.tolist()[i],
            'fbeta': fbeta.tolist()[i],
            'support': support.tolist()[i],
        }

    report['_macro'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'fbeta': macro_fbeta,
        'support': macro_support
    }

    report['_micro'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'fbeta': micro_fbeta,
        'support': micro_support
    }

    if print_output:
        print_class_report_fbeta(report)

    return report


def print_class_report_fbeta(report):
    tbl = StringTable(separator=" ")
    keys = list(report.keys())
    keys.remove("_macro")
    keys.remove("_micro")
    keys.append("_macro")
    keys.append("_micro")

    cols = ["precision", "recall", "fbeta", "support"]

    for key in keys:
        row = [key]
        for c in cols:
            if report[key][c] is not None:
                if c == "support":
                    row.append("%5d" % report[key][c])
                else:
                    row.append("%.2f" % report[key][c])
        tbl.add_cells(row)
        tbl.new_row()

    cols.insert(0, "")

    tbl.set_headline(cols)

    print(tbl.create_table(False))


if __name__ == "__main__":
    array_y_true = np.array([[0] * 10 + [1, 2, 3, 4] + [0] * 0,
                             [0] * 9 + [1, 2, 3, 4] + [0] * 1,
                             [0] * 9 + [1, 2, 3, 4] + [0] * 1])
    array_y_pred = np.array([[0] * 10 + [1, 2, 3, 4] + [0] * 0,
                             [0] * 10 + [1, 2, 3, 4] + [0] * 0,
                             [0] * 1 + [1, 2, 3, 0] + [0] * 9])

    print(array_y_pred)
    print(array_y_true)
