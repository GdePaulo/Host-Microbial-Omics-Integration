from calendar import month_name
import imp
from operator import index
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, average_precision_score, log_loss
import loader as load
import processor as pr
import pandas as pd
import numpy as np

def runCrossValidation(x, y, model, splits=2, categorical=True):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    skf.get_n_splits(x, y)

    y_tests = []
    y_predicteds = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)
        
        y_predicted = model.predict(x_test)
        # print("before rounding", y_predicted)
        if categorical:
            y_predicted = convertPredictionToCategorical(y_predicted, y)
        # print("after rounding", y_predicted)
            
        # y_prob = model.predict_

        y_tests.append(y_test)
        y_predicteds.append(y_predicted)

    # print("generating for", y_predicteds)
    sum_report = generateClassificationReport(y_tests, y_predicteds)

    return sum_report    


def convertPredictionToCategorical(prediction, all_predictions):
    
    min_bound = min(all_predictions)
    max_bound = max(all_predictions)

    # Deals with dubious supports in classification report from large predictions
    clipped_prediction = np.clip(prediction, min_bound, max_bound)
    # Clipping is necessary otherwise large values won't get rounded, failing classification report
    rounded_prediction = np.rint(clipped_prediction)
    return rounded_prediction

def generateClassificationReport(y_tests, y_predicteds):
    sum_report = {}
    total_predictions = len(y_predicteds)
    
    for i in range(total_predictions):
        y_test = y_tests[i]
        y_predicted = y_predicteds[i]
        # print(y_predicted, y_test)
        print(f"Real:{y_test.values}\nPred:{y_predicted}\n")
        cur_report = classification_report(y_test, y_predicted, output_dict=True, zero_division=0)
        # print(cur_report, " -- sum: ", sum_report)
        for metric in ["precision", "recall", "f1-score"]:
            sum_report[metric] = sum_report.get(metric, 0) + cur_report["macro avg"][metric] / total_predictions
        # if y_predicted.nunique() == 2:
        #     sum_report["pr-auc"] = sum_report.get("pr-auc", 0) +  average_precision_score(y_test, y_predicted) / total_predictions
        #     print(f"\nScore:{average_precision_score(y_test, y_predicted)}")
        for k in cur_report.keys():
            if k == "accuracy":
                break
            sum_report[f"support-{k}"] = sum_report.get(f"support-{k}", 0) + cur_report[k]["support"] / total_predictions

    sum_report["iterations"] = total_predictions

    return sum_report



def runRandomSampling(x, y, model, categorical=True):
    
    y_tests = []    
    y_predicteds = []
    
    for i in range(50):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y, random_state=42+i)

        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)

        if categorical:
            y_predicted = convertPredictionToCategorical(y_predicted, y)

        y_tests.append(y_test)
        y_predicteds.append(y_predicted)

    sum_report = generateClassificationReport(y_tests, y_predicteds)
    return sum_report

def runExperiments(data, files, target="tumor", ps=[0, 5, 10, 20, 50], sampling="cv", selection="chi2"):
    

    for i, d in enumerate(data):
        if target == "tumor":
            d = load.attachTumorStatus(d)
            model = SVC(random_state=0)
            model_name = "svc"
        elif target == "stage":
            d = load.attachStageStatus(d)
            # model = LogisticRegression(multi_class='multinomial', max_iter=400, random_state=0)
            model = LinearRegression()
            model_name = "linreg"
    

        final_reports = None
        for c in ["COAD", "ESCA", "HNSC", "READ", "STAD"][:]:   
            
            x, y = pr.splitData(d, target=target, project=c)
            
            if selection == "linreg":
                linreg_ranked_features = pr.selectFeatures(x, y, max(ps), selection)

            for p in ps:
                if target=="tumor" and files[i] == "tcma_gen_aak_ge" and c == "READ":
                    continue
                if target=="stage" and files[i] == "tcma_gen_aak_ge" and c == "READ":
                    continue

                print(f"Running {files[i]} {target} {c} {p} {sampling} {selection}")

                # Every class must have at least 2 samples
                # Also, there must be at least two classes for prediction 
                val_counts = y.value_counts(ascending=True)
                least_class = val_counts.iloc[0]

                if least_class < 2 or len(val_counts) < 2:
                    print(f"Skipping {files[i]} {c} {p} {len(x)} due to {least_class} least class")
                    continue
                
                if selection == "chi2":
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
                    best_indices = pr.selectFeatures(x_train, y_train, p, selection)
                    
                    x_selected = x_test.iloc[:, best_indices].copy()
                    y_selected = y_test
                    print(f"Running for {files[i]} {c} {p} | found p using: {len(x_train)} tr/ev using: {len(x_test)}")
                elif selection == "linreg":

                    if p == 0:
                        x_selected = x.copy()
                    else:
                        best_indices = linreg_ranked_features[:p]
                        x_selected = x.iloc[:, best_indices].copy()
                    
                    y_selected = y

                if sampling == "cv":
                    cur_report = runCrossValidation(x_selected, y_selected, model=model)
                elif sampling=="random_sampling":
                    cur_report = runRandomSampling(x_selected, y_selected, model=model)


                cur_report["cancer"] = c
                cur_report["p"] = p
                cur_report["sampling"] = sampling
                cur_report["model"] = model_name
                cur_report["selection"] = selection

                # Convert elements to array to avoid issues with lack of index when using scaler values from dictionary
                cur_report = {k : [cur_report[k]] for k in cur_report}
                report_d = pd.DataFrame.from_dict(cur_report, orient="columns")

                if final_reports is not None:
                    final_reports = pd.concat([final_reports, report_d], ignore_index=True)
                else:
                    final_reports = report_d
        base_file_name = fr'Data\Descriptor\Prediction_Tables\{sampling}\{target}\{files[i]}_{selection}_pred'
        load.createDirectory(base_file_name)
        pretty_report_file_name = base_file_name + '.txt'
        report_file_name = base_file_name + '.csv'
        final_reports.to_csv(report_file_name, index=None)
        pt = load.getPrettyTable(final_reports)
        load.saveDescriptor(pt, pretty_report_file_name)