import loader as load
import os

prediction_targets = ["tumor", "stage"]
prediction_bounds = {
    "tumor" : [0, 1],
    "stage" : [0, 4]
}

sampling = ["random_sampling"]#["cv", "random_sampling"]
random_sampling_iterations = 2#00
random_sampling_training_portion = 0.8

selection_types = ["linreg", "chi2", "elasticnet"]
# feature_amounts = [0, 6, 12]
feature_amounts = [0, 6, 10, 26, 50, 100, 200]
# feature_amounts = [0, 5, 10, 25, 50, 100, 200]
modality_parities = ["imparity", "parity"]

model_state_path = "aidata/model_state"

visualization_packages = {
    "all": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge(parity)", "tcma_gen_aak_ge_ae"],
    "base": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge'],
    "base&parity": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge(parity)"],
    "base&ae": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge_ae"],
    "stadstage": ['aak_ge', 'tcma_gen', 'tcma_gen_aak_ge', "tcma_gen_aak_ge_ae"],
}

modality_file_name_to_name = {
    'aak_ge': "GE",
    'tcma_gen': "GENUS",
    'tcma_gen_aak_ge': "GE ∩ GENUS",
    'tcma_gen_aak_ge(parity)': "GE ∩ GENUS (parity)",
    'tcma_gen_aak_ge_ae': "GE ∩ GENUS (ae)",
}

all_features, _ = load.getFeatures()
tcma_gen_features, aak_ge_features = all_features
tcma_gen_aak_ge_features = aak_ge_features + tcma_gen_features
tcma_gen_aak_ge_ae_features = list(range(30))

modality_features = {
    "aak_ge": aak_ge_features,
    'tcma_gen': tcma_gen_features,
    'tcma_gen_aak_ge': tcma_gen_aak_ge_features,
    'tcma_gen_aak_ge_ae': tcma_gen_aak_ge_ae_features,
}

# Maybe also tune normalize and tol
model_hyperparameter_ranges = {
    "ElasticNet": {
        "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
        "l1_ratio" : [step/10 for step in range(0, 10)]
     }
}

model_hyperparameter_scoring = {
    "stage":"neg_root_mean_squared_error"
}

PREDICTIONS_DIR = os.path.join("Data","Descriptor","Prediction_Tables")