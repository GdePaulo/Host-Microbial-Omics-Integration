prediction_targets = ["tumor", "stage"]
prediction_bounds = {
    "tumor" : [0, 1],
    "stage" : [0, 4]
}

sampling = ["cv", "random_sampling"]
selection_types = ["linreg", "chi2"]