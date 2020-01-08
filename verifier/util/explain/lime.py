import re

from lime.lime_text import LimeTextExplainer


def detect_relevant_word_inputs(text, predict_fn, class_names,
                                remove_empty_classes=False,
                                split_exp=r'[^A-z0-9]'):

    pattern_compiled = re.compile(split_exp, flags=re.IGNORECASE)
    split_callable = lambda x: pattern_compiled.split(x)
    num_tokens = len(pattern_compiled.split(text))

    explainer = LimeTextExplainer(class_names=class_names,
                                  split_expression=split_callable)

    num_samples = num_tokens * 100

    e = explainer.explain_instance(text, predict_fn,
                                   num_samples=num_samples,
                                   labels=range(len(class_names)))

    tokens = e.domain_mapper.indexed_string.inverse_vocab

    # align actual text and tokens
    mapping = {}
    j = 0
    actual_tokens = e.domain_mapper.indexed_string.as_list
    for i, actual_token in enumerate(actual_tokens):
        if j >= len(tokens):
            break
        if actual_token == tokens[j]:
            mapping[j] = i
            j += 1

    tokens_heat = {}
    predictions = {}
    for label_idx, class_name in enumerate(class_names):
        words_and_values = [(idx, val) for idx, val in e.local_exp[label_idx] if val > 0]
        predictions[class_name] = float(e.predict_proba[label_idx])
        if len(words_and_values) == 0:
            continue
        scale_value = max([v for _, v in words_and_values]) / e.predict_proba[label_idx]
        words_and_values = [(mapping[idx], int(round(v/scale_value * 100))) for idx, v in words_and_values]
        words_and_values = [(idx, v) for idx, v in words_and_values if v > 1]
        if not remove_empty_classes or len(words_and_values) > 0:
            tokens_heat[class_name] = words_and_values

    return actual_tokens, tokens_heat, predictions
