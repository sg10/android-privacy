from verifier import config


def train_summary(eval_values, model, generator):
    #print("epochs: ", len(eval_values['loss']))

    evaluation = list(zip(model.metrics_names, eval_values))
    for e in evaluation:
        print("%20s    %.5f" % (e[0], e[1]))


def get_t2p_word_embedding_type():
    is_word2vec = "word2vec" in config.Text2PermissionClassifier.downloaded_embedding_file.lower()
    is_glove = "glove" in config.Text2PermissionClassifier.downloaded_embedding_file.lower()

    if not is_glove and not is_word2vec:
        raise RuntimeError("embedding filename contains neither word2vec nor glove as a substring!")

    return "word2vec" if is_word2vec else "glove"
