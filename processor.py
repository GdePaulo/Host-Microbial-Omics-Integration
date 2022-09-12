import os
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Visualize data with PCA, making sure it's standardized

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

import math
import numpy as np

def splitData(d, target, project=""):

    if project:
        d = d[d["project"] == project]
        d = d.drop(["project"], axis=1)

    y = d[target]
    x = d.drop([target], axis=1)

    return x, y

def getPCA(x):
    pca = PCA(n_components=2, random_state=1)
    pcs = pca.fit_transform(x)
    pc_df = pd.DataFrame(data = pcs, columns = ['PC 1', 'PC 2'])
    return pc_df

def getTSNE(x):
    tsne = TSNE(n_components=2, random_state=1)
    tsnes = tsne.fit_transform(x)
    tsne_df = pd.DataFrame(data = tsnes, columns = ['t-SNE 1', 't-SNE 2'])
    return tsne_df


def selectFeatures(x, y, k=10, method="chi2"):
    if k == 0:
        # return x.copy()
        return list(range(len(x.columns)))

    if method == "chi2":
        selector = SelectKBest(chi2, k=k)
        selector.fit(x, y)

        best_indices = selector.get_support()
    elif method == "linreg":
        model = LinearRegression()
        model.fit(x, y)

        features_with_coefficients = pd.DataFrame({"feature":x.columns,"coefficients":np.transpose(model.coef_)})
        features_with_coefficients_abs = features_with_coefficients.copy()
        features_with_coefficients_abs["coefficients"] = features_with_coefficients_abs.apply(lambda row: abs(row.coefficients), axis=1)
        
        sorted_features = features_with_coefficients_abs.sort_values("coefficients", ascending=False)
        top_features = sorted_features.head(k)["feature"].values
        best_indices = [x.columns.get_loc(c) for c in x.columns if c in top_features]

    # chi2_scores = pd.DataFrame(list(zip(x.columns, selector.scores_, selector.pvalues_)), columns=['ftr', 'score', 'pval'])
    # chi2_scores

    # kbest = np.asarray(x.columns)[selector.get_support()]
    return best_indices
    # return x.iloc[:, best_indices].copy()

def plotScatter(X, Y, titles=[], filename="", diagnostic="tumor"):
    rows = math.ceil(len(X) / 3)

    fig = plt.figure(figsize = (9,rows*3))
    
    for i in range(0, len(X)):
        ax = fig.add_subplot(rows,3,(i+1)) 
        ax.set_xlabel(X[i].columns[0], fontsize = 15)
        ax.set_ylabel(X[i].columns[1], fontsize = 15)
        title = titles[i] if titles else "PCA"
        ax.set_title(title, fontsize = 10)

        if diagnostic == "tumor":
            cdict = {0: "g", 1: "r"}
            clabel = {0: "normal", 1: "tumor"}
        else:
            cdict = {0: "g", 1: "r", 2:"b", 3: "orange"}
            clabel = {x: "Stage" + str(x+1) for x in cdict}

        for g in np.unique(Y[i]):
            ix = np.where(Y[i] == g)
            x = X[i].iloc[ix].iloc[:, 0].values
            y = X[i].iloc[ix].iloc[:, 1].values
        
            scatter = ax.scatter(x, y, c=cdict[g], label=clabel[g])
            # ax.legend(handles=scatter.legend_elements()[0], labels=labels)
        ax.grid()
    # ax.legend(loc="upper left")
    plt.legend()

    plt.tight_layout()
    if filename:
        filename += ".png"
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
             os.makedirs(directory)
        plt.savefig(filename, transparent=False, facecolor="white")

    
# if __name__ == "__main__":

    # createGEOverlappingTCMA("phylum", includeStage=True)
