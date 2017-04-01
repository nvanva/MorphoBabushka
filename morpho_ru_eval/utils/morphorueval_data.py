import os
from morpho_ru_eval.utils import converters
from morpho_ru_eval.utils import datasets
from morpho_ru_eval.utils import gikrya


class LabeledUDDataset():
    """
    TODO
    """
    TRAIN_DATASETS = ['gikrya_new_train.out.gz', 'OpenCorpora.ud.gz', 'RNC.ud.gz', 'syntagrus_full.ud.gz']
    DEV_DATASETS = ['gikrya_new_test.out.gz']
    LABELED_DATASETS = DEV_DATASETS + TRAIN_DATASETS

    def __init__(self, parts=TRAIN_DATASETS, tag_converter=None):
        self.data_dir_ = os.path.join( os.path.dirname(os.path.abspath(__file__)), '../datasets')
        self.parts_ = parts
        self.tag_converter_ = tag_converter

    def parse_line(self, line, attrs_field_num=4):
        vals = line.split('\t')
        token, lemma, pos, attrs = vals[1], vals[2], vals[3], vals[attrs_field_num]  # TODO: add lemma support
        if attrs != '_':
            attrs_dict = dict(nv.split('=') for nv in attrs.split('|'))
        else:
            attrs_dict = {}

        if self.tag_converter_:
            return token, self.tag_converter_(token, lemma, pos, attrs_dict)
        else:
            attrs_dict[gikrya.POS_ATTRNAME] = pos
            return token, attrs_dict


    def load_sentences(self, limit=None):
        """
        Loads test set of MorhoRuEval-17 in sentences format. Tuple of sentences, tags for train set.
        """
        self.length_ = {}
        all_fields = None
        for part in self.parts_:
            if limit is not None and limit <= 0:
                break
            path = os.path.join(self.data_dir_, part)
            print('Loading %s...' % part)
            if part.startswith('OpenCorpora'):
                fields = datasets.read_data(path, parse_line=lambda x: self.parse_line(x, attrs_field_num=5), limit=limit)
            else:
                fields = datasets.read_data(path, parse_line=lambda x: self.parse_line(x, attrs_field_num=4), limit=limit)
            l = len(fields[0])
            self.length_[part] = l
            if limit is not None:
                limit -= l
            print('%s: %d sentences' % (part, l))
            if all_fields is None:
                all_fields = fields
            else:
                for f, f_new in zip(all_fields, fields):
                    f.extend(f_new)
        return all_fields


    def save_sentences(self, dir, sentences, tags):
        os.makedirs(dir, exist_ok=True)
        st = 0
        for part in self.parts_:
            len = self.length_[part]
            fpath = os.path.join(dir, part + '.pred')
            print('Saving %s...' % fpath)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, 'w') as f:
                print(converters.sentences2gikrya(sentences[st:st+len], tags[st:st+len]), file=f)
            st += len


class UnlabeledUDDataset(LabeledUDDataset):
    TEST_DATASETS = ['test_set/JZ.txt.gz', 'test_set/VK.txt.gz', 'test_set/Lenta.txt.gz']

    def __init__(self, parts=TEST_DATASETS):
        super(UnlabeledUDDataset, self).__init__(parts=parts)

    def load_sentences(self, limit=None):
        return super().load_sentences(limit)[0]

    def parse_line(self, line, attrs_field_num=4):
        vals = line.split('\t')
        token = vals[1]
        return [token]  # should return sequence, bugs otherwise!!!


if __name__=='__main__':
    # for part in LabeledUDDataset.LABELED_DATASETS:
    #     print(part)
    #     ts = LabeledUDDataset([part])
    #     st = time.time()
    #     sents, tags = ts.load_sentences()
    #     print('%d sec' % (time.time() - st))
    #     print(list(zip(sents[0], tags[0])))

    ts = UnlabeledUDDataset(parts=UnlabeledUDDataset.TEST_DATASETS[:1])
    sents = ts.load_sentences()
    ts.save_sentences('./test_test', sents, [['Pos=NOUN+Case=Nom+Gender=Masc+Number=Sing' for w in s] for s in sents])
    for s in sents[:5]:
        print(s)
