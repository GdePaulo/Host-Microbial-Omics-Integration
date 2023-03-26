import loader as load
import os
from sklearn.utils.fixes import loguniform
import numpy as np

prediction_targets = ["tumor", "stage"]
prediction_bounds = {
    "tumor" : [0, 1],
    "stage" : [0, 4]
}

sampling = ["random_sampling"]#["cv", "random_sampling"]
random_sampling_iterations = 200
random_sampling_training_portion = 0.8

selection_types = ["linreg", "chi2", "elasticnet", "lasso", "anova", "pearson"]
# feature_amounts = [0, 6, 12]
feature_amounts = [0, 6, 10, 26, 50, 100, 200]
# feature_amounts = [0, 5, 10, 25, 50, 100, 200]
modality_parities = ["imparity", "parity"]

model_state_path = "aidata"

visualization_packages = {
    "all": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge(parity)", "tcma_gen_aak_ge_ae"],
    "base": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge'],
    "base&parity": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge(parity)"],
    "base&ae": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge_ae"],
    "super_base": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge'],
    "super_base&parity": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge(parity)"],
    "super_base&ae": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge_ae"],
    "super_ae": ['aak_ge_ae', 'tcma_gen_ae', "tcma_gen_aak_ge_ae"],
    "super_base&ae&nmf": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge_ae", "tcma_gen_aak_ge_nmf"],
    "super_base&nmf": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge_nmf"],
    "super_nmf": [ "tcma_gen_aak_ge_nmf", "aak_ge_nmf", "tcma_gen_nmf"],
    "super_base&super_nmf": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge_nmf", "aak_ge_nmf", "tcma_gen_nmf"],
}

visualization_package_to_experiment_name = {
    "super_base&parity": "Modality parity enforcement",
    "base&parity": "Modality parity enforcement",
    "super_base&ae": "AE integration",
    "super_base&nmf": "NMF integration",
    "super_base": "Predictive performance",
    "base": "Predictive performance",
    "super_nmf": "NMF integration per modality",
    "super_ae": "AE integration per modality",
}

modality_file_name_to_name = {
    'aak_ge': "GE",
    'tcma_gen': "GENUS",
    'tcma_gen_aak_ge': "GE ∩ GENUS",
    'tcma_gen_aak_ge(parity)': "GE ∩ GENUS (parity)",
    'tcma_gen_aak_ge_ae': "GE ∩ GENUS (ae)",
    'aak_ge_ae': "GE (ae)",
    'tcma_gen_ae': "GENUS (ae)",
    'tcma_gen_aak_ge_nmf': "GE ∩ GENUS (nmf)",
    'tcma_gen_nmf': "GENUS (nmf)",
    'aak_ge_nmf': "GE (nmf)",
    'baseline': "Baseline",
}

target_to_name = {
    "stage": "stage",
    "tumor": "normal vs tumor"
}

all_features, _ = load.getFeatures()
tcma_gen_features, aak_ge_features = all_features
tcma_gen_aak_ge_features = aak_ge_features + tcma_gen_features
tcma_gen_aak_ge_ae_features = list(range(100)) 
aak_ge_ae_features = list(range(100)) 
tcma_gen_ae_features = list(range(50)) 
tcma_gen_aak_ge_nmf_features = list(range(100)) 
aak_ge_nmf_features = list(range(100)) 
tcma_gen_nmf_features = list(range(50)) 

modality_features = {
    "aak_ge": aak_ge_features,
    'tcma_gen': tcma_gen_features,
    'tcma_gen_aak_ge': tcma_gen_aak_ge_features,
    'tcma_gen_aak_ge(parity)': tcma_gen_aak_ge_features,
    'tcma_gen_aak_ge_ae': tcma_gen_aak_ge_ae_features,
    'aak_ge_ae': tcma_gen_aak_ge_ae_features,
    'tcma_gen_ae': tcma_gen_ae_features,
    'tcma_gen_aak_ge_nmf': tcma_gen_aak_ge_nmf_features,
    'tcma_gen_nmf': tcma_gen_aak_ge_nmf_features,
    'aak_ge_nmf': tcma_gen_aak_ge_nmf_features,
}

# Maybe also tune normalize and tol
model_hyperparameter_ranges = {
    "ElasticNet": {
        "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
        "l1_ratio" : [step/10 for step in range(0, 10)]
     },
    "Lasso": {
        "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
     },
     # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    "RandomForestRegressor": {
        # "n_estimators" : [step * 200 for step in range(1, 11)],
        "n_estimators" : [5, 20, 50, 100, 200, 400],
        # Number of features to consider at every split
        "max_features" : ['auto', 'sqrt'],
        # Maximum number of levels in tree
        # "max_depth" : [step * 10 for step in range(1, 11)],
        "max_depth" : [10, 30, 60, 100],
        # Minimum number of samples required to split a node
        "min_samples_split" : [2, 5, 10],
        # Minimum number of samples required at each leaf node
        "min_samples_leaf" : [1, 2, 4],
        # Method of selecting samples for training each tree
        "bootstrap" : [True, False]
    }, 
    "SVC": [
        {
            'C': np.logspace(-10, 10, 21),
            'gamma': np.logspace(-10, 10, 21).tolist()+['scale', 'auto'],
            'kernel': ['rbf'],
            'class_weight':['balanced', None]
        },
        {
            'C': np.logspace(-10, 10, 21),
            'kernel': ['linear'],
            'class_weight':['balanced', None]
        }]
}

model_hyperparameter_scoring = {
    "stage":"neg_root_mean_squared_error"
}

PREDICTIONS_DIR = os.path.join("Data","Descriptor","Prediction_Tables")

# https://www.datylon.com/blog/data-visualization-for-colorblind-readers#:~:text=The%20first%20rule%20of%20making,out%20of%20these%20two%20hues.
# https://personal.sron.nl/~pault/
colorblind_colors = ['#377eb8',
                      '#ff7f00',
                    '#e41a1c',

                      '#984ea3',
                    '#a65628',
                  '#f781bf',
                  '#999999',

                        '#4daf4a',
                     
                      '#dede00']

def getLegendPosition(target, metric):
    if target=="stage" and metric != "rmse":
        return "upper right"
    else:
        return "lower right"

stad_precision = 113 / (113 + 9)
baselines = {
    "stage": {
        "STAD": ((4*9 + 19 + 0 + 27 + 4*16)/107)**0.5
    },
    # 2 * (p * r) / (p + r)
    "tumor": {
        "STAD": 2 * (stad_precision * 1) / (stad_precision + 1)
    }
}