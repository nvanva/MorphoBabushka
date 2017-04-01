# coding=utf-8
__author__ = 'Oleg'
import gzip, codecs


def read_data(fpath, parse_line, limit=None):
    """
    :param fpath: plain text file or gz
    :param parse_line: function receiving line and returning row tuple (token, pos, tag for instance)
    :return: tuple of columns, each column is in sentences format (list of lists of values)
    """
    if fpath.endswith('gz'):
        with gzip.open(fpath, 'rb') as raw_inp:
            inp = codecs.getreader("utf-8")(raw_inp)
            return read_data_stream(inp, parse_line, limit=limit)
    else:
        with open(fpath, "r", encoding='utf-8') as inp:
            return read_data_stream(inp, parse_line, limit=limit)


def read_data_stream(f, parse_line, limit=None):
    fields = None  # each element of this list will corresponds to a specific field of dataset (words, tag, lemma, etc.)
    sent_rows = []
    for line in f:
        line = line.strip('\n')
        # Empty line is sentence delimiter in Gikrya, OpenCorpora, Syntagrus, RNC. Also RNC have special lines with meta data.
        if line == '' or line == '==newfile==' or line.startswith('==>') and line.endswith('<=='):
            if len(sent_rows) > 0:
                # zip(*l) convert iterable over rows to the iterable over columns
                for field, sent_field in zip(fields, (list(x) for x in zip(*sent_rows))):
                    field.append(sent_field)
                sent_rows = []
                if limit is not None and len(fields[0]) >= limit:
                    break
        else:
            parsed = parse_line(line)
            if fields is None:
                fields = [[] for _ in parsed]  # number of fileds is the number of elements returned by parse_line for the first line
            else:
                assert len(fields) == len(parsed), 'parse_line returned %d fields for 1-st line and %d fields for other line' % (len(fields), len(parsed))
            sent_rows.append(parsed)  # sent_rows is a list of rows

    return tuple(fields)


def read_gold_standard(f, attrs2tag=None):
    """
    Возвращает список списков слов (список предложений) и список списков атрибутов (список атрибутов по предложениям)
    при atr_flg=False атрибуты - просто теги pos
    при atr_flg=True атрибуты это строки вида Atr1=Value|Atr2=Value... # TODO: we have agreed to use '+' instead of '|' earlier to distinguish from initial attrs list
    Такие строки содержат Pos= как 1 из атрибутов, содержат Atr=_ если атрибут принимает значение по умолчанию, упорядочены по имени атрибута
    пример -
    'Case=Nom|Degree=Pos|Gender=Masc|Number=Sing|Pos=ADJ|Variant=_'
    'Pos=PUNCT|_' - случай если атрибутов кроме части речи нет # TODO: why this is a special case and we add '_' ?
    """

    sents = read_data(f)
    # TODO: this is lambda and map overuse and very unclear code! if you want to assign lambda to variable, you should define an ordinary function!
    # TODO: more effictively and clear to use simple 'for' and 'append' here to form words, lemms, etc. whithout intermediate representations
    s_splited = [[e.split('\t') for e in s] for s in sents]

    fnc1= lambda lst: list(map(lambda x: x[1],lst))  # word
    fnc2= lambda lst: list(map(lambda x: x[2],lst))  # lemma
    fnc3= lambda lst: list(map(lambda x: x[3],lst))  # POS
    fnc4= lambda lst: list(map(lambda x: x[4].strip('\n'),lst))  # atrib

    words = list(map(fnc1,s_splited))  # TODO: list comprehension: words=[[s[1] for elem in s] for s in s_splited]
    atrib = None

    if attrs2tag != None:
        lemms = list(map(fnc2, s_splited))
        pos_tags = list(map(fnc3, s_splited))
        atrib = list(map(fnc4, s_splited))

        for i in range(0,len(atrib)):  # TODO: cycling over list elements can be done simplier: 'for a in atrib:'
            for j in range(0,len(atrib[i])):  # TODO: use list comprehension
                attr_str = atrib[i][j]
                attr_dict = dict(nv.split('=') for nv in attr_str.split('|') if nv != '_')
                atrib[i][j] = attrs2tag(words[i][j], lemms[i][j], pos_tags[i][j], attr_dict)

    return words, atrib

def _is_last(s):
    win_size = len(s)
    end='END'
    num_pad = win_size//2

    temp = s.copy()
    temp.reverse()
    return temp[:num_pad] == [end]*num_pad

def _is_first(s):
    win_size = len(s)
    start='START'
    num_pad = win_size//2

    temp = s.copy()
    return temp[:num_pad] == [start]*num_pad

def windows2sentence(windows,win_tags):
    win_size = len(windows[0])
    num_pad = win_size//2

    cur_sent = []
    cur_tag=[]
    sentences = []
    tags=[]
    w_tags = win_tags.copy()
    for win in windows:
        if _is_first(win):
            cur_sent = [win[num_pad]]
            cur_tag = [w_tags.pop(0)]
        elif _is_last(win):
            cur_sent.append(win[num_pad])
            sentences.append(cur_sent)
            cur_tag.append(w_tags.pop(0))
            tags.append(cur_tag)
        else:
            cur_sent.append(win[num_pad])
            cur_tag.append(w_tags.pop(0))

    return sentences,tags

def sentence2windows(sentences, tags=None, win_size=3):
    """
    Converts sentences dataset (list of sentences) to windows dataset (list of windows extracted from these sentences,
    each token becomes a separate example, the center of some window and token's tag - the class of this window).
    :param sentences: list of sentences, each sentence is a list of tokens
    :param tags: list of paths (each path is a list of tags) or None
    :param win_size: window size
    :return: tuple consisting of a list of windows (each window is a list of win_size tokens with target token in the
    center, padding is added as needed) and a list of tags or None (if tags were not provided)
    """
    start='START'
    end='END'
    num_pad = win_size//2
    sent_start = [start]*num_pad
    sent_end = [end]*num_pad

    windows = []
    for sent in sentences:
        s_prep = sent_start+sent+sent_end
        for i in range(len(sent)):
            windows.append(s_prep[i:i+win_size])
    if tags is not None:
        win_tags = [tag for sent in tags for tag in sent]
        return windows, win_tags
    else:
        return windows, None
