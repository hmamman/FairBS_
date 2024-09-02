import sys
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Get the absolute path to the directory where main.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../"))

from utils import helpers
from utils.ml_classifiers import CLASSIFIERS


def prepare_data(path, target_column):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_classifier(clf_name):
    return CLASSIFIERS.get(clf_name)


def train_and_evaluate(data, clf_name, dataset_name, is_dnn=False):
    config = helpers.get_data_config(dataset_name)

    X, y, input_shape, nb_classes = data()

    if y.shape[1] == nb_classes:
        y = np.argmax(y, axis=1)

    X = pd.DataFrame(X, columns=config.feature_name)
    y = pd.DataFrame(y, columns=['y'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = get_classifier(clf_name)
    if clf is None:
        raise ValueError(f"Unsupported classifier: {clf_name}")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{clf.__class__.__name__} Model's accuracy on {dataset_name} dataset is: {accuracy:.4f}")

    return clf


def save_model(model, dataset_name, clf_name=None):
    dir_path = f"models/{dataset_name}"
    os.makedirs(dir_path, exist_ok=True)

    model_path = f"{dir_path}/{clf_name}_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <dataset_name> [classifier_name]")
        sys.exit(1)

    configs = helpers.get_config_dict()

    dataset_name = sys.argv[1]
    clf_name = sys.argv[2].lower() if len(sys.argv) == 3 else None

    data_path = f"data/{dataset_name}.py"
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    config = configs[dataset_name]

    data = helpers.get_data(config.dataset_name)

    if clf_name:
        classifiers_to_train = [clf_name]
    else:
        classifiers_to_train = CLASSIFIERS.keys()
    is_dnn = False

    for clf_name in classifiers_to_train:
        try:
            model = train_and_evaluate(data, clf_name, dataset_name, is_dnn=is_dnn)
            save_model(model, dataset_name, clf_name)
        except ValueError as e:
            print(str(e))
            continue


if __name__ == "__main__":
    main()
