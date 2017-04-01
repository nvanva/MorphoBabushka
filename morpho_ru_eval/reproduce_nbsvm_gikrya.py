import argparse

from morpho_ru_eval.utils import gikrya
from morpho_ru_eval.utils.MultiOutputWrapper import MultiOutputWrapper
from morpho_ru_eval.utils.evaluation import evaluate_clf
from morpho_ru_eval.window_sklearn import Sequence2WindowsClassifier, nbsvm1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_limit',type=int, default=None)
    args = parser.parse_args()

    win, make_clf = 5, nbsvm1

    clf = MultiOutputWrapper()
    clf.add_clf(Sequence2WindowsClassifier(win, make_clf), {gikrya.POS_ATTRNAME})
    for attrname in gikrya.get_all_eval_attrs():
        clf.add_clf(Sequence2WindowsClassifier(win, make_clf), {attrname})

    evaluate_clf(clf, 'NBSVM1', only_pos=False, log_to_file=False, train_limit=args.train_limit, predict_test=True)


if __name__ == '__main__':
    main()

