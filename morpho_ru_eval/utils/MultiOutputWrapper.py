from time import time
import morpho_ru_eval.utils.gikrya as gk


class MultiOutputWrapper():
    def __init__(self):
        self.clfs = []
        self.ExtraFit = {}
        self.ExtraPredict = {}

    def add_clf(self, clf, d, useExtra = False, saveToExtra = False, namesForUseInExtra='none', nameForSaveToExtra='none'):
        print(d, ':', clf)
        self.clfs.append([clf, d, useExtra, saveToExtra, namesForUseInExtra, nameForSaveToExtra])

    def fit(self, sents_train, tags_train):
        self.single_value_clfs = {}

        for i, (clf, attrset, useExtra, saveToExtra, namesForUseInExtra, nameForSaveToExtra) in enumerate(self.clfs):
            y_train = []
            for path in tags_train:
                y_t = []
                for tag in path:
                    d = gk.tag2dict(tag)[1]
                    if d is not None:
                        new_tag = {n:v for n, v in d.items() if n in attrset}
                        y_t.append(gk.dict2tag(pos=None, d=new_tag))
                    else:
                        y_t.append("")
                y_train.append(y_t)

            tagset = {tag for path in y_train for tag in path} - {''}  # all tags except empty tag
            if len(tagset) < 2:  # No need to train classifier for less than 2 classes!
                single_value = next(iter(tagset)) if len(tagset) == 1 else ''
                print(attrset, ':', tagset, ' - single value shortcut, no need to fit classifier', clf)
                self.single_value_clfs[i] = single_value
            elif useExtra:
                print(attrset, ':', tagset, 'fitting classifier', clf, 'with Extra info')  # TODO: more info about Extra
                st = time()
                clf.fit(sents_train, tags_train, self.ExtraFit, namesForUseInExtra)
                print('Done in %d sec' % (time() - st))
            else:
                print(attrset, ':', tagset, 'fitting classifier', clf, 'without Extra info')
                st = time()
                clf.fit(sents_train, y_train)
                print('Done in %d sec' % (time() - st))

            if saveToExtra:
                print('Saving to ExtraFit: ', nameForSaveToExtra)
                pred = self.predict_ith_clf(sents_train, i, True)
                self.ExtraFit[nameForSaveToExtra] = pred
        return self


    def predict_ith_clf(self, sents, i, fromFit=False):
        (clf, attrset, useExtra, saveToExtra, namesForUseInExtra, nameForSaveToExtra) = self.clfs[i]

        if i in self.single_value_clfs:
            single_value = self.single_value_clfs[i]
            print(attrset, ':', 'single value shortcut - predicting "%s"' % single_value)
            res = [[single_value for _ in sent] for sent in sents]
        else:
            print(attrset, ':', 'predicting using', clf)
            st = time()
            if useExtra:
                res = clf.predict(sents, self.ExtraPredict, namesForUseInExtra)
            else:
                res = clf.predict(sents)
            print('Done in %d sec' % (time()-st))
        if (saveToExtra & (not fromFit)):
            print('Saving to ExtraPredict: ', nameForSaveToExtra)
            self.ExtraPredict[nameForSaveToExtra] = res
        return res


    def predict(self, sents_test):
        paths = None
        for i, (clf, attrset, useExtra, saveToExtra, namesForUseInExtra, namesForSaveToExtra) in enumerate(self.clfs):
            pred_paths = self.predict_ith_clf(sents_test, i)

            if paths is None:
                paths = pred_paths
            else:
                print('Merging predicted...')
                paths = [[tag if pred_tag=='' else pred_tag if tag=='' else pred_tag + '+' + tag
                          for pred_tag, tag in zip(pred_p, p)] for pred_p, p in zip(pred_paths, paths)]
                print('Merging done.')
        return paths
         

def main():
	print("not implemented")

if __name__ == '__main__':
    main()


