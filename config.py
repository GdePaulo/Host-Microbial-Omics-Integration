prediction_targets = ["tumor", "stage"]
prediction_bounds = {
    "tumor" : [0, 1],
    "stage" : [0, 4]
}

sampling = ["random_sampling"]#["cv", "random_sampling"]
random_sampling_iterations = 200
random_sampling_training_portion = 0.8

selection_types = ["linreg", "chi2", "elasticnet"]
# feature_amounts = [0, 6, 12]
feature_amounts = [0, 6, 10, 26, 50, 100, 200]
# feature_amounts = [0, 5, 10, 25, 50, 100, 200]
modality_parities = ["imparity", "parity"]

model_state_path = "aidata/model_state"