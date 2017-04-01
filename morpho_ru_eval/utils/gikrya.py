import os
from morpho_ru_eval.utils import datasets
from morpho_ru_eval.utils import official_evaluation
from morpho_ru_eval.utils.morphorueval_data import LabeledUDDataset

POS_ATTRNAME = 'Pos'
ATTR_NAME_VAL_DELIM = '='
ATTR_DELIM = '+'
EMPTY_VAL = '_'

# TODO: NOUN == PROPN (convert PROPN to NOUN?)
# TODO: Variant=Brev == Variant=Short (convert Brev to Short?)
# TODO: replace 'yo' to 'ye'?
def get_all_eval_attrs():
    return {attr for pos in official_evaluation.POS_TO_MEASURE for attr in get_eval_attrs(pos)}


def get_eval_attrs(true_pos):
    return official_evaluation.get_cats_to_measure(true_pos)

def is_evaled(true_pos, token):
    return true_pos in official_evaluation.POS_TO_MEASURE or (true_pos == "ADV" and token.lower() not in official_evaluation.DOUBT_ADVERBS)


def tag2dict(tag):
    """
    Convert tag (string) representation to POS string + attrubutes dictionary representation.
    """
    if tag=='':
        return None, None
    d = dict(x.split(ATTR_NAME_VAL_DELIM) for x in tag.split(ATTR_DELIM))
    pos = d.get(POS_ATTRNAME)
    return pos, d


def dict2tag(pos=None, d=None):
    """
    Convert from POS string + attrubutes dictionary representation to tag (string) representation - name=value pairs
    separated by ATTR_DELIM and ordered alphabetically, POS is always first.
    """
    filtered = []
    if d is not None:
        filtered.extend(['%s=%s' % (n, v) for n, v in d.items()])
        filtered.sort()
    if pos is not None:
        filtered.insert(0, POS_ATTRNAME + ATTR_NAME_VAL_DELIM + pos)
    return ATTR_DELIM.join(filtered)


def filter_true_d_official(true_pos, true_d, token):
    """
    Remove from true tag elements which do not effect official evaluation (some POSes, some morphological attrs).
    Doesn't work with predicted tag because true tag determines which elements from predicted tag are used.
    :param true_pos: POS string
    :param true_d: dictionary containing attributes
    :param token: string containing token, needed to check if this token is evaluated
    :return: dictionary containing evaluated attributes
    """
    if not is_evaled(true_pos, token):
        return None
    evaled_attrs = get_eval_attrs(true_pos)
    filtered_d = {name: val for name, val in true_d.items() if name in evaled_attrs and val!=EMPTY_VAL}
    return filtered_d


# def filter_true_tag_official(true_tag, token):
#     true_pos, true_d = tag2dict(true_tag)
#     filtered_d = filter_true_d_official(true_pos, true_d, token)
    # if filtered_d is None:
    #     return ''
    # return dict2tag(true_pos, filtered_d)


def filter_tags_official(true_tag, pred_tag, token):
    true_pos, true_d = tag2dict(true_tag)
    true_filtered_d = filter_true_d_official(true_pos, true_d, token)
    if true_filtered_d is None:
        return '', ''  # Classifier is correct independent of the predicted tag
    pred_pos, pred_d = tag2dict(pred_tag)
    if pred_d is None:
        return dict2tag(true_pos, true_filtered_d), ''
    # Retain only those predicted attributes which are specified in true_tag and are evaluated
    pred_filtered_d = {name: pred_d[name] for name in true_filtered_d.keys() if name in pred_d}
    return dict2tag(true_pos, true_filtered_d), dict2tag(pred_pos, pred_filtered_d)


def filter_tags_official_sent(X, y_true, y_pred):
    y_true_filtered, y_pred_filtered = [], []
    for true_path, pred_path, sent in zip(y_true, y_pred, X):
        pairs = [filter_tags_official(true_tag, pred_tag, token) for true_tag, pred_tag, token in zip(true_path, pred_path, sent)]
        y_true_filtered.append([x for x, y in pairs])
        y_pred_filtered.append([y for x, y in pairs])
    return y_true_filtered, y_pred_filtered


def get_parts(parts='dev'):
    # parts = ['unofficial_split/gikrya_%s.txt.gz' % part]  # unofficial split used before official split appeared; use it only to compare with previous results!
    if parts=='dev':
        return LabeledUDDataset.DEV_DATASETS
    elif parts=='train':
        # return LabeledUDDataset.TRAIN_DATASETS
        return [LabeledUDDataset.TRAIN_DATASETS[0]]
    elif type(parts)==str:
        return [parts]
    else:
        return parts


def pos_sentences(parts='dev', only_evaluated=False, limit=None):
    """
    Loads GIKRYA with POS tags only.
    :param parts: train /dev part
    :param only_evaluated: return empty strings for non-evaluated POSes, filter non-evaluated attributes; results in
        fewer classes (easier to fit), but less supervised information provided to the classifier
    :return: dataset in sentences format
    """
    def pos2tag(token, lemma, pos, attrs_dict):
        if only_evaluated:
            pos = pos if is_evaled(pos, token) else None
        return dict2tag(pos, None)

    ds = LabeledUDDataset(parts=get_parts(parts), tag_converter=pos2tag)
    words, tags = ds.load_sentences(limit=limit)
    return words, tags


def posattrs_sentences(parts='dev', only_evaluated=False, limit=None):
    """
    Loads GIKRYA with tags composed of POS and morphological attributes.
    :param parts: train /dev part
    :param only_evaluated: return empty strings for non-evaluated POSes, filter non-evaluated attributes; results in
        fewer classes (easier to fit), but less supervised information provided to the classifier
    :return: dataset in sentences format
    """

    def attrs2tag(token, lemma, pos, attr_dict):
        if only_evaluated:
            attr_dict = filter_true_d_official(pos, attr_dict, token)
            if attr_dict is None:
                return ''

        return dict2tag(pos, attr_dict)

    ds = LabeledUDDataset(parts=get_parts(parts), tag_converter=attrs2tag)
    words, tags = ds.load_sentences(limit=limit)
    return words, tags


