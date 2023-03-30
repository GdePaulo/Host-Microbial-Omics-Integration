import os
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Visualize data with PCA, making sure it's standardized

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2, f_classif, r_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso
from sklearn.svm import SVC
from tuner import getTunedModel
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

# To do: fix chi2 to return list Sorted based on feature importance
# https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/
def selectFeatures(x, y, k=10, method="chi2", random_seed=0, scoring="n/a"):
    feature_coefficients = {}
    if k == 0:
        # return x.copy()
        return list(range(len(x.columns))), feature_coefficients

    if method in ["chi2", "anova", "pearson"]:
        if method == "chi2":
            selector = SelectKBest(chi2, k=k)
        elif method == "anova":
            selector = SelectKBest(f_classif, k=k)
        elif method == "pearson":
            selector = SelectKBest(r_regression, k=k)
        X_kbest = selector.fit(x, y)
        # print("e", X_kbest)
        best_indices = selector.get_support()
        # Convert boolean mask to list of indices
        best_indices = [i for i, selected in enumerate(best_indices) if selected==True]
    else:
        if method == "linreg":
            model = LinearRegression()
            model.fit(x, y)
        else:
            if method == "elasticnet":
            # Allow for the random state to be equal to the predictor models in predictor.py
                model = ElasticNet(random_state=0)
            elif method == "lasso":
                model = Lasso(random_state=0)
            elif method == "rfreg":
                model = RandomForestRegressor(random_state=0)
            print(f"Tuning selector model")
            tuned_object = getTunedModel(model, x, y, random_state=random_seed, scoring=scoring)
            model = tuned_object.best_estimator_
        coefficients = model.feature_importances_ if method == "rfreg" else model.coef_
        features_with_coefficients = pd.DataFrame({"feature":x.columns,"coefficients":np.transpose(coefficients)})
        features_with_coefficients_abs = features_with_coefficients.copy()
        features_with_coefficients_abs["coefficients"] = features_with_coefficients_abs.apply(lambda row: abs(row.coefficients), axis=1)
        
        sorted_features = features_with_coefficients_abs.sort_values("coefficients", ascending=False)
        top_features = sorted_features.head(k)["feature"].values
        feature_coefficients = sorted_features.set_index("feature").T.to_dict("records")[0]
        best_indices = [x.columns.get_loc(c) for c in top_features]

    # chi2_scores = pd.DataFrame(list(zip(x.columns, selector.scores_, selector.pvalues_)), columns=['ftr', 'score', 'pval'])
    # chi2_scores

    # kbest = np.asarray(x.columns)[selector.get_support()]
    return best_indices, feature_coefficients
    # return x.iloc[:, best_indices].copy()

def plotScatter(X, Y, sub_titles=[], filename="", diagnostic="tumor", cols=3, main_title="PLOT"):
    rows = math.ceil(len(X) / cols)

    fig = plt.figure(figsize = (cols * 3, rows * 3))
    
    for i in range(0, len(X)):
        ax = fig.add_subplot(rows,cols,(i+1)) 

        if (i == 0):
            ax.set_xlabel(X[i].columns[0], fontsize = 10)
            ax.set_ylabel(X[i].columns[1], fontsize = 10)
        title = sub_titles[i] if sub_titles else "PCA"
        ax.set_title(title, fontsize = 12)

        if diagnostic == "tumor":
            cdict = {0: "b", 1: "r"}
            clabel = {0: "normal", 1: "tumor"}
        else:
            cdict = {0: "#D81B60", 1: "#1E88E5", 2:"#FFC107", 3: "#004D40", 4: "#FE6100"}
            clabel = {x: "Stage " + str(x+1) for x in cdict}

        for g in np.unique(Y[i]):
            ix = np.where(Y[i] == g)
            x = X[i].iloc[ix].iloc[:, 0].values
            y = X[i].iloc[ix].iloc[:, 1].values
        
            scatter = ax.scatter(x, y, c=cdict[g], label=clabel[g])
            # ax.legend(handles=scatter.legend_elements()[0], labels=labels)
        ax.grid()
    # ax.legend(loc="upper left")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.suptitle(main_title, size=15)

    plt.tight_layout()
    if filename:
        filename += ".png"
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
             os.makedirs(directory)
        plt.savefig(filename, transparent=False, facecolor="white")

    
# if __name__ == "__main__":

    # createGEOverlappingTCMA("phylum", includeStage=True)
