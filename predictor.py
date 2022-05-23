from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def runCrossValidation(x, y, splits=2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    model = SVC()
    # model.fit(x_train, y_train)


    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    skf.get_n_splits(x, y)
    reports, sum_report = [], {}

    for train_index, test_index in skf.split(x, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)
        # print(y_predicted)
        # print(accuracy_score(y_test, y_predicted))
        # print(y_test, y_predicted)
         
        cur_report = classification_report(y_test, y_predicted, output_dict=True)
        # print("error?")
        for metric in ["precision", "recall", "f1-score"]:
            sum_report[metric] = sum_report.get(metric, 0) + cur_report["macro avg"][metric] / splits
        sum_report["support-normal"] = sum_report.get("support-normal", 0) + cur_report["0"]["support"] / splits
        sum_report["support-tumor"] = sum_report.get("support-tumor", 0) + cur_report["1"]["support"] / splits
        sum_report["accuracy"] = sum_report.get("accuracy", 0) + cur_report["accuracy"] / splits
    return sum_report    