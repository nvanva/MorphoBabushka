import pandas as pd
from morpho_ru_eval.utils import datasets
from morpho_ru_eval.utils import gikrya
import regex

UPPER_LETTER = regex.compile(r"\p{Lu}")
LOWER_LETTER = regex.compile(r"\p{Ll}")

def capitalization_optimized3(word):
    """
    1.5/2.3ms
    :param word:
    :return:
    """
    if word == '':
        return 4
    if word == 'START':
        return 5
    if word == 'END':
        return 6

    if UPPER_LETTER.search(word) is None:
        return 0
    elif UPPER_LETTER.search(word[1:]) is None:
        return 1
    elif LOWER_LETTER.search(word) is None:
        return 2
    else:
        return 3


def format_line(i, sent, path):
    pos, d = gikrya.tag2dict(path[i])
    if pos is None:
        pos, attrs = '_', ['_']
    else:
        attrs = ['%s=%s' % (n, v) for n, v in d.items()]
        attrs.sort()
    return '%d\t%s\t%s\t%s' % (i+1, sent[i], pos, '|'.join(attrs))


def sentences2gikrya(sents, tags):
    """
    Convert sentences representation to string in gikrya which can be saved to file and used for official evaluation script.
    :param sents:
    :param tags:
    :return:
    """
    s = '\n\n'.join('\n'.join(format_line(i, sent, path) for i in range(len(sent))) for sent, path in zip(sents, tags))
    return s


def sentences2df(sentences, paths=None, win=5, filter_empty=True, nopadding=False, Extra=None):
    wins, paths = datasets.sentence2windows(sentences, paths, win_size=win)
    df = pd.DataFrame.from_records(wins, columns=['%s' % i for i in range(-(win//2),(win//2)+1)])
    # calculate capitalization features (! do this before before addig ^ and $ !)
    cap = df.applymap(capitalization_optimized3 )
    if not nopadding:
        df.replace('^','^', inplace=True, regex=True)
        df.replace('$','$', inplace=True, regex=True)
    for col in cap.columns:
        df[col + '_cap'] = cap[col]
    if Extra:
        for name, data in Extra.items(): #TODO: Extra may contain some other data, check if it's needed (use namesForUseInExtra)
            print('converts ', name)
            wins_extra, _ = datasets.sentence2windows(data, None, win_size=win)
            df_extra = pd.DataFrame.from_records(wins_extra, columns=[str(i) + '_Extra_' + str(name) for i in range(-(win // 2), (win // 2) + 1)])
            assert len(df_extra)==len(df), 'number of tokens in sentences (%d) != number of tags in Extra (%d)' % (len(df_extra), len(df))
            df = pd.concat([df, df_extra], axis=1)
        df.head(25).to_csv('q.csv')  # TODO: remove, this is for debug only
    if paths:
        df['y_true'] = paths
    if paths and filter_empty:
        df = df[df.y_true!='']
    return df
