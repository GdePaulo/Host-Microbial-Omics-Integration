from calendar import month_name
import imp
from importlib.metadata import metadata
from operator import index

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, average_precision_score, log_loss
from tuner import getTunedModel
import loader as load
import processor as pr
import pandas as pd
import numpy as np
import config
import os

def convertPredictionToCategorical(prediction, all_predictions):
    # Hardcode
    min_bound = min(all_predictions)
    max_bound = max(all_predictions)

    # Deals with dubious supports in classification report from large predictions
    clipped_prediction = np.clip(prediction, min_bound, max_bound)
    # Clipping is necessary otherwise large values won't get rounded, failing classification report
    rounded_prediction = np.rint(clipped_prediction).astype(np.int32)
    return rounded_prediction

def generateClassificationReport(y_tests, y_predicteds):
    sum_report = {}
    total_predictions = len(y_predicteds)
    
    for i in range(total_predictions):
        y_test = y_tests[i]
        y_predicted = y_predicteds[i]
        print(f"Real:{y_test.values}\nPred:{y_predicted}\n")
        cur_report = classification_report(y_test, y_predicted, output_dict=True, zero_division=0)
        for metric in ["precision", "recall", "f1-score"]:
            sum_report[metric] = sum_report.get(metric, 0) + cur_report["macro avg"][metric] / total_predictions
        
        for k in cur_report.keys():
            if k == "accuracy":
                break
            sum_report[f"support-{k}"] = sum_report.get(f"support-{k}", 0) + cur_report[k]["support"] / total_predictions

    sum_report["iterations"] = total_predictions

    return sum_report

def generatePredictionsDataFrame(y_tests, y_predicteds, y_predicteds_raw):
    predictions = {}
    all_predicted_labels = [y for x in y_predicteds for y in x]
    all_predicted_raw_labels = [y for x in y_predicteds_raw for y in x]
    all_actual_labels = [y for x in y_tests for y in x]
    all_sampling_iterations = [i for i, x in enumerate(y_tests) for y in range(len(x))]

    predictions["predicted"] = all_predicted_labels
    predictions["predicted-raw"] = all_predicted_raw_labels
    predictions["actual"] = all_actual_labels
    predictions["iteration"] = all_sampling_iterations
    return predictions
    
def generateSelectedFeaturesDataFrame(selected_features, coefficients):
    features = {}
    all_selected_features = [y for x in selected_features for y in x]
    all_selected_features_coefficients = [y for x in coefficients for y in x]
    all_sampling_iterations = [i for i, x in enumerate(selected_features) for y in range(len(x))]
    features["features"] = all_selected_features
    features["coefficient"] = all_selected_features_coefficients
    features["iteration"] = all_sampling_iterations
    return features

# used for preloading of feature selection. Ensures the p at which which features are preloaded
# is not bigger than the total amount of available features for the current modality.
def isMaximumSelectableP(feature_row, p):
    max_p = max(config.feature_amounts)
    total_features = len(feature_row.columns)    
    if total_features < max_p:
        ps_smaller_than_total_features = [p_candidate for p_candidate in config.feature_amounts if p_candidate < total_features]
        return p == max(ps_smaller_than_total_features)
    else:
        return p == max_p

loaded_features = {}
loaded_features_coefficients = {}
def runRandomSampling(x, y, model, categorical=True, selection="chi2", p=0, preload_features=True, modality_selection_parity=False, hyperp_tuning=False, hyperp_scoring="accuracy"):
    
    y_tests = []    
    y_predicteds = []
    selected_features = []
    selected_features_coefficients = []
    
    global loaded_features
    global loaded_features_coefficients
    if preload_features and p==max(config.feature_amounts):
        loaded_features = {}
        loaded_features_coefficients = {}
    
    if modality_selection_parity:
        all_features, _ = load.getFeatures()
        gen, ge = all_features
        p_per_modality = round(p/2)

    for i in range(config.random_sampling_iterations):
        iteration_seed = 42+i
        # print(f"Random sampling iteration {i}")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - config.random_sampling_training_portion), stratify=y, random_state=iteration_seed)
        

        # print("Selecting features")
        if modality_selection_parity:
            if isMaximumSelectableP(x, p) or not preload_features:
                x_train_with_genus_features = x_train[gen]
                x_train_with_ge_features = x_train[ge]

                best_indices_genus, coefficients_genus = pr.selectFeatures(x=x_train_with_genus_features, y=y_train, k=p_per_modality, method=selection, random_seed=iteration_seed, scoring=hyperp_scoring)
                best_indices_ge, coefficients_ge = pr.selectFeatures(x=x_train_with_ge_features, y=y_train, k=p_per_modality, method=selection, random_seed=iteration_seed, scoring=hyperp_scoring)

                # Do it like this to preserve order
                best_features_genus = [gen[best_index] for best_index in best_indices_genus]
                best_features_ge = [ge[best_index] for best_index in best_indices_ge]
                # best_features_both_modalities = best_features_genus + best_features_ge

                best_indices_genus_original = [x_train.columns.get_loc(f) for f in best_features_genus]
                best_indices_ge_original = [x_train.columns.get_loc(f) for f in best_features_ge]

                best_indices = best_indices_genus_original + best_indices_ge_original
                coefficients = coefficients_genus | coefficients_ge
                
                if preload_features:
                    loaded_features[i] = (best_indices_genus_original, best_indices_ge_original)
                    loaded_features_coefficients[i] = coefficients
            elif p==0:
                best_indices, coefficients = pr.selectFeatures(x=x_train, y=y_train, k=p, method=selection, random_seed=iteration_seed, scoring=hyperp_scoring)
            else:
                if preload_features:
                    best_indices_genus, best_indices_ge = loaded_features[i]
                    best_indices = best_indices_genus[:p_per_modality] + best_indices_ge[:p_per_modality]
                    coefficients = loaded_features_coefficients[i]
        else:
            if isMaximumSelectableP(x, p) or not preload_features:
                best_indices, coefficients = pr.selectFeatures(x=x_train, y=y_train, k=p, method=selection, random_seed=iteration_seed, scoring=hyperp_scoring)
                if preload_features:
                    loaded_features[i] = best_indices
                    loaded_features_coefficients[i] = coefficients
            elif p==0:
                best_indices, coefficients = pr.selectFeatures(x=x_train, y=y_train, k=p, method=selection, random_seed=iteration_seed, scoring=hyperp_scoring)
            else:
                if preload_features:
                    best_indices = loaded_features[i][:p]
                    coefficients = loaded_features_coefficients[i]
                

        x_train_selected = x_train.iloc[:, best_indices].copy()  
        x_test_selected = x_test.iloc[:, best_indices].copy()  

        
        if hyperp_tuning:
            print("Hyperparameter tuning model")
        # print("Fitting model")
            predictor = getTunedModel(model, x_train_selected, y_train, random_state=iteration_seed, scoring=hyperp_scoring)
            best_parameters = predictor.best_params_
            print(f"Finished tuning param: {best_parameters}")
        else:
            model.fit(x_train_selected, y_train)
            predictor = model

        # print("Done fitting model")
        y_predicted = predictor.predict(x_test_selected)

        # print("before rounding", y_predicted)
        # print("after rounding", y_predicted)

        y_tests.append(y_test)
        y_predicteds.append(y_predicted)
        
        if p != 0:
            selected_feature_names = [x.columns[best_index] for best_index in best_indices]
            selected_feature_coefficients = [coefficients.get(feature_name, -1) for feature_name in selected_feature_names]
            selected_features.append(selected_feature_names)
            selected_features_coefficients.append(selected_feature_coefficients)

    return y_tests, y_predicteds, selected_features, selected_features_coefficients

def runExperiments(data, files, target="tumor", ps=config.feature_amounts, sampling="cv", selection="chi2", modality_selection_parity=False, stad_exp=False, selected_model="no"):
    categorical = True

    for i, d in enumerate(data):

        if stad_exp:
            print("Running special STAD-STAGE experiments")
            
        if target == "tumor":
            d = load.attachTumorStatus(d)
            scoring = "balanced_accuracy"
        elif target == "stage":
            d = load.attachStageStatus(d)
            # model = LogisticRegression(multi_class='multinomial', max_iter=400, random_state=0)
            scoring = "neg_root_mean_squared_error"

        if selected_model == "ElasticNet":
            model = ElasticNet(random_state=0)
            model_name = "enet"
        elif selected_model == "SVC":
            model = SVC(random_state=0)
            model_name = "svc"
        elif selected_model == "LinearRegression":
            model = LinearRegression()
            model_name = "linreg"
        elif selected_model == "RandomForestRegressor":
            model = RandomForestRegressor(random_state=0)
            model_name = "rfreg"
            

        preload_features = True
        if selection in ["chi2", "anova", "pearson"]:
            # Do not preload because they are not Sorted when using chi2
            preload_features = False
        enforce_modality_parity = modality_selection_parity
        # Don't enforce modalities if there is only one modality
        if modality_selection_parity and files[i] != "tcma_gen_aak_ge":
            enforce_modality_parity = False

        final_reports = [None, None, None]
        for c in ["COAD", "ESCA", "HNSC", "READ", "STAD"][:]:   
            if stad_exp and (target!="stage" or c!="STAD"):
                continue

            x, y = pr.splitData(d, target=target, project=c)
            
            for p in reversed(ps):
                if target=="tumor" and files[i] == "tcma_gen_aak_ge" and c == "READ":
                    continue
                if target=="stage" and files[i] == "tcma_gen_aak_ge" and c == "READ":
                    continue
                nr_columns = len(x.columns)
                if nr_columns <= p:
                    print(f"Skipping prediction iteration for {files[i]} as p:{p}>={nr_columns}")
                    continue

                print(f"Running {files[i]} {target} {c} {p} {sampling} {selection}")

                # Every class must have at least 2 samples
                # Also, there must be at least two classes for prediction 
                val_counts = y.value_counts(ascending=True)
                
                least_class = val_counts.iloc[0]

                if least_class < 2 or len(val_counts) < 2:
                    print(f"Skipping {files[i]} {c} {p} {len(x)} due to {least_class} least class")
                    continue
                
                # print(f"Running for {files[i]} {c} {p}")
                if sampling=="random_sampling":
                    y_tests, y_predicteds, selected_features, selected_features_coefficients = runRandomSampling(x, y, model=model, selection=selection, p=p, preload_features=preload_features, modality_selection_parity=enforce_modality_parity, hyperp_tuning=True, hyperp_scoring=scoring)
                
                if categorical:
                    y_predicteds_clipped_and_rounded = [convertPredictionToCategorical(y_predicted, y) for y_predicted in y_predicteds]
                    print("Generating classification report")
                    cur_report = generateClassificationReport(y_tests, y_predicteds_clipped_and_rounded)
                
                print("Generating predictions dataframe")
                cur_pred_output_report = generatePredictionsDataFrame(y_tests, y_predicteds_clipped_and_rounded, y_predicteds)
                print("Generating selected features dataframe")
                cur_pred_features_report = generateSelectedFeaturesDataFrame(selected_features, selected_features_coefficients)
                
                metadata = {
                    "cancer" : c,
                    "p" : p,
                    "sampling" : sampling,
                    "model" : model_name,
                    "selection" : selection
                } 
                reports_to_save = [cur_report, cur_pred_output_report]

                if p!=0:
                    reports_to_save.append(cur_pred_features_report)

                for j, report in enumerate(reports_to_save):
                    
                    for k, v in metadata.items():
                        if report == cur_report:
                            report[k] = v
                        else:
                            first_column_values = list(report.values())[0]
                            report[k] = [v for _ in range(len(first_column_values))]

                    # Convert elements to array to avoid issues with lack of index when using scaler values from dictionary
                    if report == cur_report:
                        report = {k : [report[k]] for k in report}
                    
                    report_d = pd.DataFrame.from_dict(report, orient="columns")

                    if final_reports[j] is not None:
                        final_reports[j] = pd.concat([final_reports[j], report_d], ignore_index=True)
                    else:
                        final_reports[j] = report_d

        prediction_performances, prediction_outputs, prediction_features = final_reports
        parity = "(parity)" if enforce_modality_parity else "" 
        super = "super" if stad_exp else "" 
        base_file_name = os.path.join(config.PREDICTIONS_DIR,super,sampling,target,model_name,selection,f"{files[i]}_{selection}{parity}_pred")
        load.createDirectory(base_file_name)

        pretty_report_file_name = base_file_name + '.txt'
        report_file_name = base_file_name + '.csv'
        predictions_output_file_name = base_file_name + "output" + ".csv"
        predictions_feature_file_name = base_file_name + "feature" + ".csv"

        prediction_outputs.to_csv(predictions_output_file_name, index=None)
        prediction_features.to_csv(predictions_feature_file_name, index=None)
        prediction_performances.to_csv(report_file_name, index=None)

        pt = load.getPrettyTable(prediction_performances)
        load.saveDescriptor(pt, pretty_report_file_name)