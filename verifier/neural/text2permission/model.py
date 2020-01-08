import keras
from keras.layers import *

from verifier import config
from verifier.preprocessing.pretrained_embedding import PreTrainedEmbeddings


def model_multiconv_1d(num_permissions):
    embedding_dim = config.Text2PermissionClassifier.embedding_dim
    sequence_length = config.Text2PermissionClassifier.max_description_embeddings

    input_layer = Input(shape=(None,))

    conv_layers = []
    for filter_size in config.Text2PermissionClassifier.conv_filters_sizes:
        conv_layer_i = Embedding(input_dim=PreTrainedEmbeddings.get().embedding_matrix.shape[0],
                                 output_dim=embedding_dim,
                                 input_length=sequence_length,
                                 weights=[PreTrainedEmbeddings.get().embedding_matrix],
                                 trainable=False)(input_layer)
        conv_layer_i = Conv1D(filters=config.Text2PermissionClassifier.conv_filters_num,
                              kernel_size=filter_size,
                              padding='same',
                              activation='relu')(conv_layer_i)
        conv_layer_i = GlobalMaxPooling1D()(conv_layer_i)

        conv_layers.append(conv_layer_i)

    if len(conv_layers) == 1:
        previous_layer = conv_layers[0]
    else:
        concatenated_layer = concatenate(conv_layers, axis=-1)
        previous_layer = concatenated_layer

    for n_neurons in config.Text2PermissionClassifier.dense_layers:
        previous_layer = Dense(n_neurons, activation='relu')(previous_layer)
        previous_layer = Dropout(config.Text2PermissionClassifier.dropout)(previous_layer)

    output_layer = Dense(num_permissions, activation='sigmoid')(previous_layer)

    return keras.Model(inputs=input_layer, outputs=output_layer)
