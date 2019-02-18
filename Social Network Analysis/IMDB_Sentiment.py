#IMPORTS
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    splitDoc = []
    if keep_internal_punct == True:
        newDoc = doc.lower().split()
        for w in newDoc:
            if not all(c in string.punctuation for c in w):
                w = w.strip(string.punctuation)
                splitDoc.append(w)
        return np.array(splitDoc)
    else:
        splitDoc = re.sub('\W+', ' ', doc.lower()).split()
        return np.array(splitDoc)


def token_features(tokens, feats):
    counterList = Counter(['token=' + s for s in tokens])
    for w, c in counterList.items():
        feats[w] = c


def token_pair_features(tokens, feats, k=3):
    windows = []
    combList = []
    for i in range(0, len(tokens) - (k - 1)):
        windows.append(tokens[i : i + k])
    for w in windows:
        for c in list(combinations(w, 2)):
            combList.append(c[0] + '__' + c[1])
    counterList = Counter(['token_pair=' + s for s in combList])
    for w, c in counterList.items():
        feats[w] = c
        
neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for w in tokens:
        if w.lower() in neg_words:
            feats['neg_words'] += 1
        if w.lower() in pos_words:
            feats['pos_words'] += 1



def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for f in feature_fns:
        f(tokens, feats)
    return sorted(feats.items(),key=lambda x: x[0])


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    indptr = [0]
    indices = []
    data = []
    features = dict()
    masterlist = list()

    for token_list in tokens_list:
        sublist = defaultdict(lambda: 0)
        sublist = featurize(token_list, feature_fns)
        masterlist.append(sublist)

    if(vocab == None):
        for token_list in masterlist:
            for feat in token_list:
                key = feat[0]
                value = feat[1]
                if value > 0:
                    if key in features:
                        features[key] = features[key] + 1
                    else:
                        features[key] = 1

        vocabdict = {key: value for key, value in features.items() if value>=min_freq}

        if(vocabdict):
            vocab = dict(vocabdict)
            index = 0
            for voc in sorted(vocab):
                vocab[voc]= index
                index = index + 1

            for doc in masterlist:
                for feat in sorted(doc):
                    if feat[1] > 0:
                        if feat[0] in vocab:
                            indices.append(vocab[feat[0]])
                            data.append(feat[1])
                indptr.append(len(indices))

            x = csr_matrix((data, indices, indptr), dtype='int64')
            return x, vocab
        return None, None
    else:
        for doc in masterlist:
            for feat in sorted(doc):
                if feat[1] > 0:
                    if feat[0] in vocab:
                        indices.append(vocab[feat[0]])
                        data.append(feat[1])
            indptr.append(len(indices))

        x = csr_matrix((data, indices, indptr), dtype='int64')
        return x, vocab




def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(n_splits = k, shuffle = False, random_state = 42)
    accuracies = []
    for train_ind, test_ind in cv.split(X):
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    dictList = []
    allFeats = []
    for i in range(len(feature_fns)):
        for j in combinations(feature_fns, i + 1):
            allFeats.append(list(j))
    for punct in punct_vals:
        tokens_list = [tokenize(d, punct) for d in docs]
        for freqs in min_freqs:
            for features in allFeats:
                    X, vocab = vectorize(tokens_list, features, freqs, None)
                    accuracies = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                    dic ={}
                    dic['punct']= punct
                    dic['features']= features
                    dic['min_freq']= freqs
                    dic['accuracy']= accuracies
                    dictList.append(dic)
    return sorted(dictList, key = lambda x : x['accuracy'], reverse = True)

def plot_sorted_accuracies(results):
    accuracy = []
    for i in results:
        accuracy.append(i['accuracy'])
    sortedAccuracy = sorted(accuracy)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.plot(sortedAccuracy)
    plt.savefig('accuracies.png')



def mean_accuracy_per_setting(results):
    meanList = []
    count = defaultdict(list)
    for i in results:
        f_name = 'features=' + ' '.join(i.__name__ for i in i['features'])
        p_name = 'punct=' + str(i['punct'])
        m_name = 'min_freq=' + str(i['min_freq'])
        count.setdefault(f_name,[])
        count[f_name].append(i['accuracy'])
        count.setdefault(p_name,[])
        count[p_name].append(i['accuracy'])
        count.setdefault(m_name,[])
        count[m_name].append(i['accuracy'])
    for k, v in count.items():
        meanAccuracy = np.mean(v)
        meanList.append((meanAccuracy, k))
    return sorted(meanList, key=lambda x: x[0], reverse = True)

def fit_best_classifier(docs, labels, best_result):
    punct = best_result['punct']
    feature_fns = best_result['features']
    min_freq = best_result['min_freq']
    tokens_list = []
    for i in docs:
        tokens_list.append(tokenize(i, punct))
    X, vocab = vectorize(tokens_list, feature_fns, min_freq, vocab = None)
    clf= LogisticRegression().fit(X,labels)
    return (clf, vocab)

def top_coefs(clf, label, n, vocab):
    coefList = []
    coef = clf.coef_[0]
    idx2word = dict((v,k) for k,v in vocab.items())
    if label == 1:
        # Sort them in descending order.
        topPosIdx = np.argsort(coef)[::-1][:n]
        for i in topPosIdx:
            feature_name = idx2word[i]
            coefficient = coef[i]
            coefList.append((feature_name, coefficient))
    if label == 0:
        topNegIdx = np.argsort(coef)[:n]
        for i in topNegIdx:
            feature_name = idx2word[i]
            coefficient = abs(coef[i])
            coefList.append((feature_name, coefficient))
    return coefList

def parse_test_data(best_result, vocab):
    punct = best_result['punct']
    feature_fns = best_result['features']
    min_freq = best_result['min_freq']
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    docs=[tokenize(d,punct) for d in test_docs]
    X_test, vocab=vectorize(docs, feature_fns, min_freq, vocab)
    return (test_docs, test_labels, X_test)


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    proba = clf.predict_proba(X_test)
    predict = clf.predict(X_test)
    topMcf = []
    for i in range(len(test_labels)):
        #both predict and test_labels can only be 1 or 0
        if predict[i] != test_labels[i]:
            if predict[i] == 0:
                topMcf.append((test_labels[i],predict[i],proba[i][0],test_docs[i]))
            else:
                topMcf.append((test_labels[i],predict[i],proba[i][1],test_docs[i]))
    sortedTopMcf = sorted(topMcf,key=lambda x: -x[2])
    for i in sortedTopMcf[:n]:
        print("\ntruth=%d predicted=%d proba=%.6f\n%s" % (i[0], i[1], i[2], i[3]))

def main():
    feature_fns = [token_features, token_pair_features, lexicon_features]
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
