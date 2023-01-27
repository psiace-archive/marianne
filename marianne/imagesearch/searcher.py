"""The image searcher for marianne"""
# marianne/imagesearch/searcher.py

import csv

import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import pearsonr, spearmanr


class Searcher:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def search(self, queryFeatures, method, limit=10):
        # initialize our dictionary of results
        results = {}
        # open the index file for reading
        with open(self.indexPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)
            # loop over the rows in the index
            for row in reader:
                # parse out the image ID and features, then compute the
                # chi-squared distance between the features in our index
                # and our query features
                features = [float(x) for x in row[1:]]
                if method == "chi2":
                    d = self.chi2_distance(features, queryFeatures)
                elif method == "euclidean":
                    d = self.euclidean_distance(features, queryFeatures)
                elif method == "manhattan":
                    d = self.manhattan_distance(features, queryFeatures)
                elif method == "chebyshev":
                    d = self.chebyshev_distance(features, queryFeatures)
                elif method == "hamming":
                    d = self.hamming_distance(features, queryFeatures)
                elif method == "cosine":
                    d = self.cosine_similarity(features, queryFeatures)
                elif method == "pearson":
                    d = self.pearson_similarity(features, queryFeatures)
                elif method == "spearman":
                    d = self.spearman_similarity(features, queryFeatures)
                elif method == "jaccard":
                    d = self.jaccard_similarity(features, queryFeatures)
                elif method == "mse":
                    d = self.mse_similarity(features, queryFeatures)
                else:
                    print("Sorry, we don't support this method.")
                    exit(1)

                # now that we have the distance between the two feature
                # vectors, we can update the results dictionary -- the
                # key is the current image ID in the index and the
                # value is the distance we just computed, representing
                # how 'similar' the image in the index is to our query
                results[row[0]] = d
            # close the reader
            f.close()
        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])
        if (
            method == "pearson"
            or method == "cosine"
            or method == "spearman"
            or method == "mse"
        ):
            results.sort(reverse=True)
        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum(
            [((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)]
        )
        # return the chi-squared distance
        return d

    def euclidean_distance(self, histA, histB):
        d = np.sum([(a - b) ** 2 for (a, b) in zip(histA, histB)]) ** 0.5
        return d

    def manhattan_distance(self, histA, histB):
        d = np.sum(np.abs(a - b) for (a, b) in zip(histA, histB))
        return d

    def chebyshev_distance(self, vec1, vec2):
        npvec1, npvec2 = np.array(vec1), np.array(vec2)
        return max(np.abs(npvec1 - npvec2))

    def hamming_distance(self, inA, inB):
        d = 0
        for i in range(len(inA)):
            if inA[i] != inB[i]:
                d += 1
        return d

    def cosine_similarity(self, x, y, norm=False):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def pearson_similarity(self, x, y):
        # x_=x-np.mean(x)
        # y_=y-np.mean(y)
        # d=np.dot(x_,y_)/(np.linalg.norm(x_)*np.linalg.norm(y_))
        return pearsonr(x, y)[0]

    def spearman_similarity(self, x, y):
        return spearmanr(x, y)[0]

    def jaccard_similarity(self, x, y):
        matV = np.mat([x, y])
        return dist.pdist(matV, "jaccard")[0]

    def mse_similarity(self, line_MSEs1, line_MSEs2, Confident=0.8):
        Diff_value = np.abs(np.array(line_MSEs1) - np.array(line_MSEs2))
        fingle = np.array(Diff_value < (1 - Confident) * np.max(Diff_value)) + 0
        similar = fingle.reshape(1, -1)[0].tolist()
        similar = sum(similar) / len(similar)

        if similar == 0.0:
            similar = 1
        return
