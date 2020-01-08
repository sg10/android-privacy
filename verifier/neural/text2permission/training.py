import os
import random
import numpy as np
import keras
from keras.optimizers import Adam
from statistics import mean

from sklearn.model_selection import KFold
from tensorflow._api.v1 import logging

from verifier.neural.text2permission.model import model_multiconv_1d
from verifier.preprocessing.permissionparser import PermissionParser
from verifier.preprocessing.pretrained_embedding import PreTrainedEmbeddings
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.neural.text2permission.callbacks import PrintSamples, PrintPerClassMetrics
from verifier.neural.text2permission.datagen import Generator, EmbeddedSamples
from verifier.util import metrics
from verifier import config
from verifier.util.metrics import print_class_report_fbeta
from verifier.util.train import get_t2p_word_embedding_type


def train(verbose=True, all_folds=False):

    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
        logging.set_verbosity(logging.ERROR)

    description_embedding_tokens_type = "description_num_tokens_%s" % get_t2p_word_embedding_type()

    db = SamplesDatabase.get()
    package_names = db.filter(#('lang', '==', 'en'),
                              ('set', '==', 'train+valid'),
                              (description_embedding_tokens_type, '>=', 20))
    random.shuffle(package_names)

    if verbose:
        print("packages from db with criteria: ", len(package_names))

    k_fold_splitter = KFold(n_splits=config.Text2PermissionClassifier.validation_split)

    k_reports_valid = []
    k_reports_test = []

    package_names_test = db.filter(('lang', '==', 'en'),
                                   ('set', '==', 'test'),
                                   (description_embedding_tokens_type, '>=', 20))
    random.shuffle(package_names_test)
    test_generator = Generator(packages=package_names_test, batch_size=128, verbose=False)

    keras.backend.clear_session()

    model = model_multiconv_1d(PermissionParser(mode='groups').count())

    for fold_number, (train_index, valid_index) in enumerate(k_fold_splitter.split(package_names)):
        print("FOLD:                 ", fold_number+1)

        packages_train = np.array(package_names)[train_index].tolist()
        packages_valid = np.array(package_names)[valid_index].tolist()

        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(0.0001),
                      metrics=[metrics.fb_micro, metrics.fb_macro, metrics.precision, metrics.recall])
        train_metric = 'val_fb_macro'

        # keras.utils.plot_model(model, "model.png", show_shapes=True)
        if verbose and fold_number == 0:
            model.summary()

        train_generator = Generator(packages=packages_train, verbose=verbose)
        valid_generator = Generator(packages=packages_valid, batch_size=128, verbose=verbose)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor=train_metric,
                                          mode='max',
                                          min_delta=config.Text2PermissionClassifier.early_stopping_delta,
                                          patience=config.Text2PermissionClassifier.early_stopping_patience,
                                          verbose=verbose),
            keras.callbacks.ModelCheckpoint(filepath=config.TrainedModels.text2permission,
                                            monitor=train_metric,
                                            mode='max',
                                            save_best_only=True,
                                            verbose=verbose)
        ]

        #if verbose:
        #    callbacks.append(PrintSamples(valid_generator, print_per_label=3))
        #    callbacks.append(PrintPerClassMetrics(valid_generator))

        model.fit_generator(train_generator,
                            epochs=config.Text2PermissionClassifier.max_train_epochs,
                            shuffle=True,
                            class_weight=permission_class_weights(packages_train),
                            validation_data=valid_generator,
                            use_multiprocessing=False,
                            verbose=verbose,
                            callbacks=callbacks
                            )

        model.load_weights(config.TrainedModels.text2permission)

        if verbose:
            print("-" * 80)
            print("-  done!")
            print("-" * 80)

            print("--- VALIDATION")

        print_metrics = PrintPerClassMetrics(valid_generator)
        print_metrics.model = model
        report_valid = print_metrics.predict_batch(print_report=verbose)

        if verbose:
            print("--- TEST")

        print_metrics = PrintPerClassMetrics(test_generator)
        print_metrics.model = model
        report_test = print_metrics.predict_batch(print_report=verbose)

        #if verbose:
        #    print_samples = PrintSamples(test_generator)
        #    print_samples.model = model
        #    print_samples.predict_batch()

        if not all_folds:
            return report_valid, report_test
        else:
            k_reports_valid.append(report_valid)
            k_reports_test.append(report_test)

    del model

    avg_reports_valid = {}
    avg_reports_test = {}

    for row in k_reports_valid[0].keys():
        for col in list(k_reports_valid[0].values())[0].keys():
            avg_reports_valid[row] = avg_reports_valid.get(row, {})
            avg_reports_valid[row][col] = mean([k_reports_valid[r].get(row, {}).get(col, 0.) or 0. for r in range(len(k_reports_valid))])
            avg_reports_test[row] = avg_reports_test.get(row, {})
            avg_reports_test[row][col] = mean([k_reports_test[r].get(row, {}).get(col, 0.) or 0. for r in range(len(k_reports_test))])

    if verbose:
        print("*" * 50)
        print(" - average over all %d folds" % k_fold_splitter.n_splits)
        print("*" * 50)
        print("VALIDATION")
        print_class_report_fbeta(avg_reports_valid)
        print()
        print("TEST")
        print_class_report_fbeta(avg_reports_test)

    return avg_reports_valid, avg_reports_test


def permission_class_weights(package_names):
    permission_parser = PermissionParser('groups')
    db = SamplesDatabase.get()

    count = np.zeros(shape=permission_parser.count())

    for p in package_names:
        count += permission_parser.transform(db.read(p, 'permissions'))

    weights = len(package_names)/(permission_parser.count() * count)
    weights = {i: w for i, w in enumerate(weights)}

    return weights


def random_search():
    batch_size = [32, 64, 128]
    conv_filters_num = [128, 256, 512, 1024]
    conv_filters_sizes = [1, 3]
    dense_layers_neurons = [100, 5000]
    dense_layers_num = [1, 10]
    dropout = [0.0, 0.4]

    print("starting random search ...")

    while True:
        cfg = {}
        cfg['dropout'] = random.uniform(dropout[0], dropout[1])

        n_layers = random.randint(dense_layers_num[0], dense_layers_num[1])
        neurons = [int(random.randint(dense_layers_neurons[0], dense_layers_neurons[1])/n_layers)
                       for _ in range(n_layers)]
        cfg['dense_layers'] = neurons

        cfg['dropout'] = random.uniform(dropout[0], dropout[1])

        cfg['conv_filters_sizes'] = list(range(1, random.randint(conv_filters_sizes[0]+1, conv_filters_sizes[1])))
        cfg['conv_filters_num'] = random.choice(conv_filters_num)

        cfg['batch_size'] = random.choice(batch_size)

        for key, value in cfg.items():
            setattr(config.Text2PermissionClassifier, key, value)

        #print("-" * 80)
        #pprint.pprint(cfg)

        report_valid, report_test = train(verbose=False)

        #print("valid:")
        #pprint.pprint(report_valid)
        #print("test:")
        #pprint.pprint(report_test)

        out = {
            'cfg': cfg,
            'valid': report_valid,
            'test': report_test
        }

        print(out)


def compare_embeddings():
    #config.Text2PermissionClassifier.conv_filters_sizes = [1, 2]
    #config.Text2PermissionClassifier.conv_filters_num = 1024
    #config.Text2PermissionClassifier.batch_size = 32
    #config.Text2PermissionClassifier.dense_layers = [712, 390]
    #config.Text2PermissionClassifier.dropout = 0.33

    for file in ["word2vec-wiki-news-300d-1M.vec", "glove.6B.300d.txt"]:
        keras.backend.clear_session()

        config.Text2PermissionClassifier.downloaded_embedding_file = config.data_folder + "/word_embeddings/%s" % file

        avg_reports_valid, avg_reports_test = train(verbose=False, all_folds=True)

        print("AVERAGE -> ", file)

        print("VALIDATION")
        print_class_report_fbeta(avg_reports_valid)
        print()
        print("TEST")
        print_class_report_fbeta(avg_reports_test)

        PreTrainedEmbeddings._instance = None
        EmbeddedSamples._instance = None


def final():
    avg_reports_valid, avg_reports_test = train(verbose=True, all_folds=False)

    print("VALIDATION")
    print_class_report_fbeta(avg_reports_valid)
    print()
    print("TEST")
    print_class_report_fbeta(avg_reports_test)


if __name__ == "__main__":
    #random_search()
    #compare_embeddings()
    final()
