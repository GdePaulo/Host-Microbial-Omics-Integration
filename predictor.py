import imp
from operator import index
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
import loader as load
import processor as pr
import pandas as pd

def runCrossValidation(x, y, splits=2, model="SVC"):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    if model == "SVC":
        model = SVC()
    elif model == "MLGR":
        model = LogisticRegression(multi_class='multinomial', max_iter=400)

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    skf.get_n_splits(x, y)
    sum_report = {}

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)

        cur_report = classification_report(y_test, y_predicted, output_dict=True, zero_division=0)
        # print(cur_report, " -- sum: ", sum_report)
        for metric in ["precision", "recall", "f1-score"]:
            sum_report[metric] = sum_report.get(metric, 0) + cur_report["macro avg"][metric] / splits
        
        if model != "MLGR":
            sum_report["pr-auc"] = sum_report.get("pr-auc", 0) +  average_precision_score(y_test, y_predicted) / splits
        print(f"Real:{y_test.values}\nPred:{y_predicted}\nScore:{average_precision_score(y_test, y_predicted)}")
        for k in cur_report.keys():
            if k == "accuracy":
                break
            sum_report[f"support-{k}"] = sum_report.get(f"support-{k}", 0) + cur_report[k]["support"] / splits

    return sum_report    

def runExperiments(data, files, target="tumor", ps=[0, 5, 10]):
    for i, d in enumerate(data):

        if target == "tumor":
            d = load.attachTumorStatus(d)
            model = "SVC"
        elif target == "stage":
            d = load.attachStageStatus(d)
            model = "MLGR"

        final_reports = None
        for c in ["COAD", "ESCA", "HNSC", "READ", "STAD"]:   
            for p in ps:
                if target=="tumor" and files[i] == "tcma_gen_aak_ge" and c == "READ":
                    continue
                if target=="stage" and files[i] == "tcma_gen_aak_ge" and c == "READ":
                    continue

                x, y = pr.splitData(d, target=target, project=c)

                # Every class must have at least 2 samples
                # Also, there must be at least two classes for prediction 
                val_counts = y.value_counts(ascending=True)
                least_class = val_counts.iloc[0]

                if least_class < 2 or len(val_counts) < 2:
                    print(f"Skipping {files[i]} {c} {p} {len(best_x)} due to {least_class} least class")
                    continue
                
                best_x = pr.selectFeatures(x, y, p)
                x = best_x

                print(f"Running for {files[i]} {c} {p} {len(best_x)}")
                cur_report = runCrossValidation(x, y, model=model)

                cur_report["cancer"] = c
                cur_report["cv_folds"] = 2
                cur_report["p"] = p

                # Convert elements to array to avoid issues with lack of index when using scaler values from dictionary
                cur_report = {k : [cur_report[k]] for k in cur_report}
                report_d = pd.DataFrame.from_dict(cur_report, orient="columns")

                if final_reports is not None:
                    final_reports = pd.concat([final_reports, report_d], ignore_index=True)
                else:
                    final_reports = report_d
        file_name = fr'Data\Descriptor\Prediction_Tables\{target}\{files[i]}_pred.txt'
        final_reports.to_csv(fr"Data\Descriptor\Prediction_Tables\{target}\{files[i]}_pred.csv", index=None)
        pt = load.getPrettyTable(final_reports)
        load.saveDescriptor(pt, file_name)