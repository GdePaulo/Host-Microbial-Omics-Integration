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

def splitData(d, target, project=""):

    if project:
        d = d[d["project"] == project]
        d = d.drop(["project"], axis=1)

    y = d[target]
    x = d.drop([target], axis=1)

    return x, y

def getPCA(x):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x)
    pc_df = pd.DataFrame(data = pcs, columns = ['principal component 1', 'principal component 2'])
    return pc_df

def getTSNE(x):
    tsne = TSNE(n_components=2)
    tsnes = tsne.fit_transform(x)
    tsne_df = pd.DataFrame(data = tsnes, columns = ['tsne component 1', 'tsne component 2'])
    return tsne_df


def selectFeatures(x, y, k=10):
    selector = SelectKBest(chi2, k=k)
    selector.fit(x, y)

    # chi2_scores = pd.DataFrame(list(zip(x.columns, selector.scores_, selector.pvalues_)), columns=['ftr', 'score', 'pval'])
    # chi2_scores

    # kbest = np.asarray(x.columns)[selector.get_support()]
    best_indices =  selector.get_support()
    return x.iloc[:, best_indices].copy()

def plotScatter(X, Y, titles=[], filename=""):
    fig = plt.figure(figsize = (8,8))
    
    for i in range(0, len(X)):
        ax = fig.add_subplot(3,3,(i+1)) 
        ax.set_xlabel(X[i].columns[0], fontsize = 15)
        ax.set_ylabel(X[i].columns[1], fontsize = 15)
        title = titles[i] if titles else "PCA"
        ax.set_title(title, fontsize = 10)

        colors = matplotlib.colors.ListedColormap(["g", "r"])
        
        scatter = ax.scatter(X[i].iloc[:, 0].values, X[i].iloc[:, 1].values, c=Y[i].values, cmap=colors)
        ax.legend(handles=scatter.legend_elements()[0], labels=["Normal", "Tumor"])
        ax.grid()
    plt.tight_layout()
    if filename:
        filename += ".png"
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
             os.makedirs(directory)
        plt.savefig(filename, transparent=False, facecolor="white")

    
# if __name__ == "__main__":

    # createGEOverlappingTCMA("phylum", includeStage=True)
