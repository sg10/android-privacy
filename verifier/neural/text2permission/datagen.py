import bs4
import keras
import logging
import numpy as np
import os
import pickle
import re

from verifier import config
from verifier.preprocessing.permissionparser import PermissionParser
from verifier.preprocessing.pretrained_embedding import PreTrainedEmbeddings
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.util.text_processing import tokenize_text, get_stopwords_list


class Generator(keras.utils.Sequence):

    def __init__(self, packages,
                 shuffle=True, at_once=False,
                 verbose=True,
                 batch_size=None):
        self.log = logging.getLogger()

        self.batch_size = batch_size or config.Text2PermissionClassifier.batch_size
        self.packages = packages

        self.at_once = at_once  # for validation: no batches, calculate at once

        self.permissions_parser = PermissionParser(mode='groups')
        self.num_permissions = self.permissions_parser.count()

        self.shuffle = shuffle
        self.indexes = []
        self.db = SamplesDatabase.get()

        if verbose:
            print("Generator loaded: %d files" % len(self.packages))

        self.embedded_samples = EmbeddedSamples.get()
        self.on_epoch_end()

    def __len__(self):
        if self.at_once:
            return 1

        return int(np.floor(len(self.packages) / self.batch_size))

    def __getitem__(self, index):
        X, y, _ = self.get_item_and_package(index)
        return X, y

    def get_item_and_package(self, index):
        if self.at_once:
            packages_temp = self.packages
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            packages_temp = [self.packages[k] for k in indexes]

        X, y, metas = self.__data_generation(packages_temp, True)
        return X, y, metas

    def __data_generation(self, packages_temp, return_packages=False):
        embedding_idx_unknown = PreTrainedEmbeddings.get().get_unknown_idx()

        # Initialization
        num_samples = len(self.packages) if self.at_once else self.batch_size
        X = np.full((num_samples, config.Text2PermissionClassifier.max_description_embeddings),
                    fill_value=embedding_idx_unknown,
                    dtype=np.int32)
        y = np.empty((num_samples, self.num_permissions), dtype=np.uint8)
        packages = []

        # Generate data
        for i, package in enumerate(packages_temp):
            embedding_indices = self.embedded_samples.get_embedded_indices(package)
            X[i, :len(embedding_indices)] = embedding_indices
            y[i] = self.permissions_parser.transform(self.db.read(package, 'permissions'))
            packages.append(package)

        if return_packages:
            return X, y, packages
        else:
            return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.packages))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.embedded_samples.on_epoch_end()


class EmbeddedSamples:

    _instance = None

    description_embedding_indices = {}
    compiled_pattern = None

    @staticmethod
    def get():
        if EmbeddedSamples._instance is None:
            EmbeddedSamples._instance = EmbeddedSamples()

        EmbeddedSamples._instance.checked_if_up_to_date = False

        return EmbeddedSamples._instance

    def __init__(self):
        self.db = SamplesDatabase.get()
        self.load()
        self.checked_if_up_to_date = False

        self.num_added = 0

    def load(self):
        if not os.path.exists(config.Text2PermissionClassifier.pre_embedded_samples_indices):
            print("embedded samples: could not find cached")
            self.checked_if_up_to_date = True
            return

        self.description_embedding_indices = pickle.load(
            open(config.Text2PermissionClassifier.pre_embedded_samples_indices, "rb"))

        if len(self.description_embedding_indices) == 0:
            print("embedded samples: empty")
        else:
            print("embedded samples: loading cached")

    def get_embedded_indices(self, package):
        if package not in self.description_embedding_indices:
            self.num_added += 1
            text = self.db.read(package, 'description_raw')
            indices = self.text_to_embedding_indices(text)
            self.description_embedding_indices[package] = indices
        else:
            indices = self.description_embedding_indices[package]

            if not self.checked_if_up_to_date:
                text = self.db.read(package, 'description_raw')
                indices2 = self.text_to_embedding_indices(text)
                if len([i for i, j in zip(indices, indices2) if i == j]) != len(indices):
                    print("flushed cache for description embedding indices")
                    self.description_embedding_indices = dict()
                    self.description_embedding_indices[package] = indices2
                    self.num_added = 1

                    indices = indices2

                self.checked_if_up_to_date = True

        return indices

    def on_epoch_end(self):
        if self.num_added > 0:
            pickle.dump(self.description_embedding_indices,
                        open(config.Text2PermissionClassifier.pre_embedded_samples_indices, "wb"))

            print("embedded samples: # added/miss = ", self.num_added)

        self.num_added = 0


    @staticmethod
    def text_to_embedding_indices(text):
        text_raw = bs4.BeautifulSoup(text, features="lxml").text

        if EmbeddedSamples.compiled_pattern is None:
            pattern = PreTrainedEmbeddings.get().get_delimiter_regex_pattern()
            EmbeddedSamples.compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)

        tokens = EmbeddedSamples.compiled_pattern.split(text_raw)
        tokens = [t for t in tokens if len(t) > 0 and t.isalnum()]
        tokens = filter(lambda t: t.lower() not in get_stopwords_list(), tokens)

        max_embeddings = config.Text2PermissionClassifier.max_description_embeddings
        embedding_indices = PreTrainedEmbeddings.get().tokens_to_indices(tokens, max_embeddings)

        #print(" ".join([PreTrainedEmbeddings.get().index2word(idx) for idx in embedding_indices]))

        return embedding_indices


def get_predict_fn(model):
    embedding_idx_unknown = PreTrainedEmbeddings.get().get_unknown_idx()

    def predict_fn(input_text_variations):
        #for inp in input_text_variations:
        #    print(inp)

        X = np.full((len(input_text_variations), config.Text2PermissionClassifier.max_description_embeddings),
                    embedding_idx_unknown)

        for i, s in enumerate(input_text_variations):
            embedding_indices = EmbeddedSamples.text_to_embedding_indices(s)
            X[i, :len(embedding_indices)] = embedding_indices

        y = model.predict(X, batch_size=256)

        return y

    return predict_fn
