# coding=utf-8
import os
from collections import OrderedDict

import pandas as pd
import sys
from sklearn_crfsuite import metrics

from morpho_ru_eval.utils import converters
from morpho_ru_eval.utils import gikrya
from datetime import datetime

from morpho_ru_eval.utils import official_evaluation
from morpho_ru_eval.utils import morphorueval_data


def time_str():
    return ('%s' % datetime.now().replace(microsecond=0)).replace(' ','_')


# def score(y_true, y_pred, verbose=True):
#     if verbose:
#         print(metrics.flat_classification_report(y_true, y_pred, digits=3))
#
#     return metrics.flat_accuracy_score(y_true, y_pred)


def off_acc(X, y_true, y_pred, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    true_file = run_dir + 'true.csv'
    with open(true_file, 'w') as f:
        print(converters.sentences2gikrya(X, y_true), file=f)
    pred_file = run_dir + 'pred.csv'
    with open(pred_file, 'w') as f:
        print(converters.sentences2gikrya(X, y_pred), file=f)
    quality = official_evaluation.measure_quality_files(true_file, pred_file, measure_lemmas=False)
    correct_tags, total_tags, correct_sents_by_tags = \
        quality['correct_tags'], quality['total_tags'], quality['correct_sents_by_tags']
    return correct_tags / total_tags


def score_df(X, y_true, y_pred, run_dir=None, only_filt_metrics=True):
    dd = OrderedDict()
    n = ['filt']
    v = [ gikrya.filter_tags_official_sent(X, y_true, y_pred) ]
    if not only_filt_metrics:
        n.append('full')
        v.append((y_true, y_pred))
    for prefix, (y_t, y_p) in zip(n, v):
    # for prefix, (y_t, y_p) in zip(['filt'], [gikrya.filter_tags_official_sent(X, y_true, y_pred)]):
        print(prefix)

        if run_dir:
            dd['off_acc'] = off_acc(X, y_true, y_pred, run_dir)
        else:
            dd['off_acc'] = 0.0

        labels = [x for x in set(tag for sent in y_t for tag in sent)]
        print('Classification report for labels: ', labels)
        if '' in labels:
            labels.remove('')  # Remove empty tag from evaluation
        print(metrics.flat_classification_report(y_t, y_p, labels=labels, digits=4))

        dd[prefix+'_acc'] = metrics.flat_accuracy_score(y_t, y_p)
        for avg in ['micro', 'macro', 'weighted']:
            dd.update({prefix+'_f1_'+avg: metrics.flat_f1_score(y_t, y_p, labels=labels, average=avg)})

    return pd.DataFrame(dd, index=[0])


def fit_predict_score(clf, train_X, train_y, dev_X, dev_y, run_dir, only_filt_metrics=True, score_trainset=False):
    print(time_str(), 'fit_predict_score: Fitting on TRAIN set (%d examples) %r...' % (len(train_X),clf))
    clf = clf.fit(train_X, train_y)
    print(time_str(), 'fit_predict_score: Predicting on DEV set (%d examples) %r...' % (len(dev_X),clf))
    pred = clf.predict(dev_X)
    print(time_str(), 'fit_predict_score: Scoring on DEV set %r...' % clf)
    dev_scores_df = score_df(dev_X, dev_y, pred, run_dir, only_filt_metrics=only_filt_metrics)
    if score_trainset:
        print(time_str(), 'fit_predict_score: Predicting on train set (%d examples) %r...' % (len(train_X),clf))
        train_pred = clf.predict(train_X)
        print(time_str(), 'fit_predict_score: Scoring on TRAIN set %r...' % clf)
        # run_dir=None to save time on saving results on train set to files for official script
        train_scores_df = score_df(train_X, train_y, train_pred, run_dir=None, only_filt_metrics=only_filt_metrics)
    print(time_str(), 'fit_predict_score: all done.')
    # Merge scores DataFrames for dev and train set
    scores_df = dev_scores_df
    if score_trainset:
        for col in train_scores_df.columns:
            if col.endswith('acc'):
                scores_df['TRAIN-' + col] = train_scores_df[col]
    return scores_df, pred, clf


def evaluate_clf(clf, run_label, only_pos=False, log_to_file=False, train_limit=None, train_parts='train', predict_test=False):
    """
    Evaluate classifier on GIKRYA.
    :param train_limit: use int value to limit train set (usefull for debugging - faster training, but much worse results!)
    :param log_to_file:
    :param clf: object with fit(X, y), predict(X) methods, where X,y - dataset in setences format (list of lists of tokens / tags)
    :param run_label: label of the run, used as prefix for the results file (should not contain characters not allowed in filenames)
    :param only_pos: evaluate only on POS-tags, use only for fast pre-evaluation (final classifier should also return morphological attributes)
    :return:
    """
    run_dir = './runs/%s-%s/' % (run_label, time_str())
    os.makedirs(run_dir, exist_ok=True)
    if not log_to_file:
        return evaluate_clf_(clf, only_pos, run_dir, train_limit=train_limit, train_parts=train_parts, predict_test=predict_test)

    terminal = sys.stdout
    try:
        with open(run_dir+'.log','w') as sys.stdout:
            return evaluate_clf_(clf, only_pos, run_dir, train_limit=train_limit, train_parts=train_parts, predict_test=predict_test)
    finally:
        sys.stdout = terminal


def evaluate_clf_(clf, only_pos, run_dir, train_limit=None, train_parts='train', predict_test=False):
    dfs = []
    if only_pos:
        print(time_str(), 'Loading POS gikrya...')
        train_X, train_y = gikrya.pos_sentences(parts=train_parts, only_evaluated=False, limit=train_limit)
        dev_X, dev_y = gikrya.pos_sentences(parts='dev', only_evaluated=False)
        df, pred_y, _ = fit_predict_score(clf, train_X, train_y, dev_X, dev_y, run_dir + 'POS/', only_filt_metrics=False)
        df['eval_method'] = 'POS'
        dfs.append(df)
    else:
        print(time_str(), 'Loading POS+Attrs gikrya...')
        train_X, train_y = gikrya.posattrs_sentences(parts=train_parts, only_evaluated=True, limit=train_limit)
        dev_X, dev_y = gikrya.posattrs_sentences(parts='dev', only_evaluated=True)
        df, pred_y, clf = fit_predict_score(clf, train_X, train_y, dev_X, dev_y, run_dir + 'POS+Attrs/', only_filt_metrics=True)
        df['eval_method'] = 'POS+Attrs'
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['clf'] = ('%s' % clf).replace('\n', '_')
    df.to_csv(run_dir + 'results.csv', sep='\t', index=False, float_format='%.3lf')

    if predict_test:
        classify_test_set(clf, run_dir)

    return df


def classify_test_set(clf, run_dir):
    ts = morphorueval_data.UnlabeledUDDataset()
    test_sents = ts.load_sentences()
    print(time_str(), 'classify_test_set: Predicting on TEST set (%d examples) %r...' % (len(test_sents),clf))
    pred = clf.predict(test_sents)
    ts.save_sentences(run_dir, test_sents, pred)


def test1(only_pos):
    for only_evaluated in True, False:
        X, y_true = gikrya.pos_sentences(parts='train', only_evaluated=only_evaluated) if only_pos \
            else gikrya.posattrs_sentences(parts='train', only_evaluated=only_evaluated)
        print(score_df(X, y_true, y_true))


if __name__ == '__main__':
    test1(True)
    test1(False)
