from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import config

def getTunedModel(estimator, x_inner, y_inner, random_state=42, scoring="neg_root_mean_squared_error"):
    model_name = type(estimator).__name__
    param_grid = config.model_hyperparameter_ranges[model_name]

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # grid_search = GridSearchCV(
    grid_search = RandomizedSearchCV(
        param_distributions = param_grid,
        n_iter = 100,
        estimator=estimator,
        # param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring, #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        random_state=random_state)

    grid_search.fit(x_inner, y_inner)
    
    # Predict on the test set and call accuracy
    return grid_search