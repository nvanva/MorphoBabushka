from time import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn_pandas import DataFrameMapper

from morpho_ru_eval.utils import gikrya
from morpho_ru_eval.utils.MultiOutputWrapper import MultiOutputWrapper
from morpho_ru_eval.utils.converters import sentences2df
from morpho_ru_eval.utils.evaluation import evaluate_clf
from sklearn_ext.nbsvm import nbsvm

def vec(ngram_range, lowercase, binary, min_df, max_df):
    return CountVectorizer(lowercase=lowercase, ngram_range=ngram_range, analyzer=u'char',
                           min_df=min_df, max_df=max_df, binary=binary)


def mapper(df, ngram_range, lowercase, binary, min_df=2, max_df=1.0, caps_features=True, pos_features=False):
    numbers = set('%d' % i for i in range(-100,100))
    feature_extractors = []
    feature_extractors.extend([
        (col, vec(ngram_range, lowercase, binary, min_df, max_df)) for col in df.columns if col in numbers
    ])  # we are using separate CountVectorizer for each word position (for min_df to work correctly)
    if caps_features:
        feature_extractors.extend([
            ([col], OneHotEncoder(sparse=False, handle_unknown='ignore')) for col in df.columns if col.endswith('_cap')
        ])
    if pos_features:
        feature_extractors.extend([
              (col, LabelBinarizer()) for col in df.columns if '_Extra' in col
        ])
    return DataFrameMapper(feature_extractors, sparse=True)


def pipeline(df, ngram_range, lowercase, binary, min_df=2, max_df=1.0, caps_features=False, pos_features=False, clf=LinearSVC()):
    return Pipeline([
        ('mapper', mapper(df, ngram_range, lowercase, binary, min_df, max_df, caps_features, pos_features)),
        ('clf', clf)
    ])


def tfidf_pipeline(df, ngram_range, lowercase, binary, min_df=2, max_df=1.0, caps_features=False, pos_features=False, clf=LinearSVC()):
    return Pipeline([
        ('mapper', mapper(df, ngram_range, lowercase, binary, min_df, max_df, caps_features, pos_features)),
        ('scaler', TfidfTransformer()),
        ('clf', clf),
    ])


def nbsvm_pipeline(df, ngram_range, lowercase, binary, min_df=2, max_df=1.0, caps_features=False, pos_features=False,
                   fit_scaler=None, transform_scaler='bin',
                   clf=LinearSVC()):
    return Pipeline([
        ('mapper', mapper(df, ngram_range, lowercase, binary, min_df, max_df, caps_features, pos_features)),
        ('clf', nbsvm(base_clf=clf, fit_scaler=fit_scaler, transform_scaler=transform_scaler) )
    ])


class Sequence2WindowsClassifier(BaseEstimator, ClassifierMixin):
    """
    For each word predicts the most frequent tag occured for this word in train set. If word didn't occure in train set,
    predicts the most frequent tag overall.
    """
    def __init__(self, winsize, make_windows_clf, nopadding=False):
        self.winsize = winsize
        self.make_windows_clf = make_windows_clf
        self.nopadding = nopadding


    def fit(self, sentences, paths, Extra=None, namesForUseInExtra=None):
        X, _ = self.to_df(sentences, paths, Extra)
        self.base_clf = self.make_windows_clf(X)
        print('Fitting on train set of %d windows...' % len(X))
        st = time()
        pipe = self.base_clf.fit(X, X.y_true)
        if hasattr(pipe, 'get_params'):
            params = pipe.get_params()
            if 'mapper' in params:
                tmp = params['mapper'].transform(X.head())
                print('Number of features from mapper: ', tmp.shape[-1])

        print('Fitting done in %d sec.' % (time() - st))
        return self

    def to_df(self, sentences, paths, Extra=None):
        print('Converting to DataFrame of windows...')
        n_tokens = 0
        for sent in sentences:
            n_tokens += len(sent)
        X_train = sentences2df(sentences, paths, win=self.winsize,
                               filter_empty=True, nopadding=self.nopadding, Extra=Extra)  # skip empty labels (don't train on them)
        print('%d sentences with %d tokens converted to %d windows.' % (len(sentences), n_tokens, len(X_train)))
        return X_train, n_tokens

    def predict(self, sentences, Extra=None, namesForUseInExtra=None):
        if(Extra != None):
            print("sentences ", len(sentences))
            print("extra ", len(Extra['Pos']))
        X, n_tokens = self.to_df(sentences, paths=None, Extra=Extra)
        assert len(X) == n_tokens, 'In predict the number of windows (%d) should match then number of tokens (%d)!' % (len(X), n_tokens)
        print('Predicting on %d windows...' % len(X))
        st = time()
        y = self.base_clf.predict(X)
        assert len(y) == n_tokens
        it = iter(y)
        paths = [[next(it) for _ in sent] for sent in sentences]
        print('Predicting done in %d sec.' % (time() - st))
        return paths

    def __str__(self):
        return "%s(winsize=%d, base_clf=%r)" % (self.__class__.__name__, self.winsize, self.make_windows_clf)


def nbsvm1(X_train):
    return nbsvm_pipeline(X_train, (1, 5), lowercase=True, binary=False, min_df=3, max_df=1.0, caps_features=True, pos_features=False,
                          fit_scaler=None, transform_scaler='bin', clf=LinearSVC(C=0.04))

def main():
    win, make_clf = 5, nbsvm1

    clf = MultiOutputWrapper()
    clf.add_clf(Sequence2WindowsClassifier(win, make_clf), {gikrya.POS_ATTRNAME})
    for attrname in gikrya.get_all_eval_attrs():
        clf.add_clf(Sequence2WindowsClassifier(win, make_clf), {attrname})

    # evaluate_clf(clf, 'MNB_Wrappers@only_evaluated', only_pos=False, log_to_file=True, only_evaluated=True)

    evaluate_clf(clf, 'MNB_Wrappers', only_pos=False, log_to_file=True)

    # clf = Sequence2WindowsClassifier(win, make_clf)
    # evaluate_clf(clf, 'Sequence2WindowsClassifier_win%d_%s@only_evaluated' % (win, make_clf.__name__),only_pos=True, log_to_file=False, only_evaluated=True)
        # evaluate_clf(clf, 'MemoryBaseline_lower=True@full',log_to_file=False, only_evaluated=False)

if __name__ == '__main__':
    main()

