prediction_targets = ["tumor", "stage"]
prediction_bounds = {
    "tumor" : [0, 1],
    "stage" : [0, 4]
}

sampling = ["random_sampling"]#["cv", "random_sampling"]
random_sampling_iterations = 100
random_sampling_training_portion = 0.8

selection_types = ["linreg", "chi2"]
feature_amounts = [0, 5, 10, 25, 50, 100, 200]