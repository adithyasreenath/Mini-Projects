#IMPORTS
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    tokenList = []
    for i in movies['genres']:
        tokenList.append(tokenize_string(i))
    movies['tokens'] = tokenList
    return movies

def featurize(movies):
    termCount = Counter()
    for i in movies['tokens']:
        termCount.update(set(i))
    vocab = {term : num for num, term in enumerate(sorted(termCount))}
    column_nums = len(vocab)
    matrixList = []
    N = len(movies)
    for i in movies['tokens']:
        termCount1 = Counter(i)
        total = sum(termCount1.values())
        max_k = max(termCount1.values())
        tfidf = []
        for term, tf in termCount1.items():
            df = termCount[term]
            tfidf.append(tf / max_k * math.log10(N / df))
        row = [0] * len(termCount1)
        col = []
        for i in termCount1:
            col.append(vocab[i])
        matrixList.append(csr_matrix((tfidf, (row, col)), shape=(1, column_nums)))
    movies['features'] = matrixList
    return(movies, vocab)



def train_test_split(ratings):
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    return np.dot(a, b.T).toarray()[0][0] / (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))




def make_predictions(movies, ratings_train, ratings_test):
    result = []
    for i, row in ratings_test.iterrows():
        iFeature = movies.loc[movies['movieId'] == row['movieId']].squeeze()['features']
        trainMovie = ratings_train.loc[ratings_train['userId'] == row['userId']]
        cosList = []
        cosSum = 0
        for i1, row1 in trainMovie.iterrows():
            tFeature = movies.loc[movies['movieId'] == row1['movieId']].squeeze()['features']
            cosSim = cosine_sim(iFeature, tFeature)
            if cosSim > 0:
                cosList.append(cosSim * row1['rating']);
                cosSum += cosSim
        if cosSum > 0:
            result.append(sum(cosList) / cosSum)
        else:
            result.append(trainMovie['rating'].mean()) 
    return np.array(result)


def mean_absolute_error(predictions, ratings_test):
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
