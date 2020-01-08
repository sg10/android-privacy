from keras import backend as K
import numpy as np


np.seterr(divide='ignore', invalid='ignore')


class HeatmapCalculator:

    def __init__(self, model, conv_layers_names):
        self.model = model
        self.conv_layers_names = conv_layers_names
        self.iterate_functions = {}

        self.init_iterate_functions()

    def init_iterate_functions(self):
        for i_output in range(self.model.output.shape[1]):
            for conv_layer_name in self.conv_layers_names:
                key = self.iterate_function_key(conv_layer_name, i_output)

                class_output = self.model.output[:, i_output]
                last_conv_layer = self.model.get_layer(conv_layer_name)
                if last_conv_layer.strides[0] != 1:
                    raise RuntimeError("heatmap calculation only works with stride=1")
                grads = K.gradients(class_output, last_conv_layer.output)[0]
                pooled_grads = K.mean(grads, axis=(0, 1))
                iterate_function = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])

                self.iterate_functions[key] = iterate_function

    def iterate_function_key(self, conv_layer_name, i_output):
        key = "%s-%d" % (conv_layer_name, i_output)
        return key

    def create(self, class_idx, x):
        input_length = x.shape[1]
        sum_heatmap = None

        for conv_layer_name in self.conv_layers_names:
            # heatmap shift b/c of padding
            feature_map_length = self.model.get_layer(conv_layer_name).output_shape[1]
            if input_length != feature_map_length:
                raise RuntimeError("input length != feature map length: %d / %d" % (input_length, feature_map_length))

            iterate = self.iterate_functions.get(self.iterate_function_key(conv_layer_name, class_idx))
            pooled_grads_value, conv_layer_output_value = iterate([x])
            #print(conv_layer_name)
            #print(feature_map_length)
            #print("pooled grads ", pooled_grads_value.shape)
            #print(pooled_grads_value)
            #print("conv_layer_output_value ", conv_layer_output_value.shape)
            #print(conv_layer_output_value)

            n_maps = pooled_grads_value.shape[0]

            for i in range(n_maps):
                conv_layer_output_value[:, i] *= pooled_grads_value[i]

            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = np.maximum(heatmap, 0)

            if sum_heatmap is None:
                sum_heatmap = heatmap
            else:
                sum_heatmap = sum_heatmap + np.maximum(sum_heatmap, heatmap)

        sum_heatmap /= np.max(sum_heatmap)
        sum_heatmap = np.nan_to_num(sum_heatmap)

        return sum_heatmap


def text_with_heatmap(words, heatmap, words_highlight=[]):
    while len(heatmap) < len(words):
        heatmap.append(0.)

    line_targets_max = []
    line_heats = []
    line_words = []
    for i, word in enumerate(words):
        x = "x" if word in words_highlight else "-"
        m = "m" if i == np.argmax(heatmap[:len(words)]) else " "
        line_targets_max.append("%s %s" % (x, m))
        line_heats.append("%.2f" % heatmap[i])
        line_words.append(word)

    lines_string = []
    for line in [line_targets_max, line_heats, line_words]:
        lines_string.append(" ".join([("%10s " % col)[:10] for col in line]))

    return lines_string


def top_words_in_heatmap(words, heatmap, n=5, threshold=0.):
    idx_sorted = np.argsort(-heatmap[:len(words)])
    idx_sorted = idx_sorted[:n]
    return [(words[idx], heatmap[idx]) for idx in idx_sorted if heatmap[idx] > threshold]
