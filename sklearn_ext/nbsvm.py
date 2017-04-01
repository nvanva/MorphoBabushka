from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import Pipeline

__author__ = 'test'

from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

class MinMaxScalerSparse(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        q=X.max(axis=0).toarray().flatten()
        q[q==0.0]=1.0
        r=np.divide(np.ones(q.shape,float),q)
        self._recimax = csr_matrix(r)
        return self

    def transform(self, X):
        return X.multiply(self._recimax)


class MNBScaler(BaseEstimator, TransformerMixin):
    '''
    ! Supports only binary problems, doesn't support multiclass ! For multiclass classifier use sklearn_ext() function.
    Multiplies each feature value with the log ration of probabilities to meet this feature in positive and negative
    classes. Current implementation works only for binary classification.
    Inspired by the paper
    [Sida Wang and Chris Manning. Baselines and Bigrams: Simple, Good Sentiment and Text Classification, 2012]
    where NBSVM (Linear SVM with NB features) is introduced.
    MNBScaler(fit_scaler=None, transform_scaler='bin')+LogisticRegression reproduces results of alternative implementation
    of NBSVM (https://github.com/mesnilgr/sklearn_ext) which is better than original implementation on IMDB Dataset (91.56% ).
    MNBScaler(fit_scaler='bin', transform_scaler='bin')+LogisticRegression shows good results (91.63%) on IMDB reviews sentiment dataset
    (better than other scalers or no scaler at all).
    '''
    fit_scalers = {None: None, 'bin': Binarizer, 'minmax': MinMaxScalerSparse}
    transform_scalers = {None: None, 'bin': Binarizer, 'auto': None}


    def __init__(self, fit_scaler=None, transform_scaler='bin'):
        self.fit_scaler=fit_scaler
        self.transform_scaler=transform_scaler
        if fit_scaler in MNBScaler.fit_scalers:
            self.fit_scaler_ = None if fit_scaler is None else MNBScaler.fit_scalers[fit_scaler]()
        else:
            raise ValueError("fit_scaler should be one of %r but %s specified" %
                             (MNBScaler.fit_scalers.keys(), fit_scaler))

        if transform_scaler in MNBScaler.transform_scalers:
            self.transform_scaler_ = None if transform_scaler is None else \
                             self.fit_scaler_ if transform_scaler=='auto' else \
                            MNBScaler.transform_scalers[transform_scaler]()
        else:
            raise ValueError("transform_scaler should be one of %r but %s specified" %
                             (MNBScaler.transform_scalers.keys(), transform_scaler))
        self.mnb_ = MultinomialNB()


    def fit(self, X, y=None):
        scaler = self.fit_scaler_
        X_scaled = X if scaler is None else scaler.fit_transform(X, y)
        mnb = self.mnb_.fit(X_scaled, y)
        self.r_ = csr_matrix(mnb.feature_log_prob_[1] - mnb.feature_log_prob_[0])
        return self


    def transform(self, X):
        scaler = self.transform_scaler_
        X_scaled = X if scaler is None else scaler.transform(X)
        return X_scaled.multiply(self.r_)


def nbsvm(base_clf, fit_scaler=None, transform_scaler='bin', multi_class='ovr'):
    """
    NB-SVM classifier: pipeline of MNBScaler+base_clf wrapped in OneVsRestClassifier / OneVsOneClassifier to support
    multiclass (MNBScaler supports only binary problems itself!).
    :param base_clf: classifier to use after MNBScaler, LogisticRegression or LinearSVC are usually used
    :param fit_scaler: look at MNBScaler class
    :param transform_scaler: look at MNBScaler class
    :param multi_class: ovr for OneVsRestClassifier, ovo for OneVsOneClassifier
    :return: OneVsRestClassifier / OneVsOneClassifier
    """
    mnb_scaler = MNBScaler(fit_scaler=fit_scaler, transform_scaler=transform_scaler)
    pipe = Pipeline([('mnbscaler', mnb_scaler), ('clf', base_clf)])
    if multi_class=='ovr':
        return OneVsRestClassifier(pipe)
    elif multi_class=='ovo':
        return OneVsOneClassifier(pipe)
    else:
        raise ValueError('Unsuppported multi_class=%s, should be one of %r.' % (multi_class, ['ovr','ovo']))