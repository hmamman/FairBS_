import os
import sys


base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../"))

import argparse
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.cluster import KMeans

from data.bank import bank_data
from data.census import census_data
from data.compas import compas_data
from data.credit import credit_data
from data.meps import meps_data
from utils.config import census, bank, credit, compas, meps


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--dataset_name', type=str, default='census', help='Name of the dataset')
    parser.add_argument('--sensitive_name', type=str, default='age',
                        help='Name of the sensitive parameter (e.g., sex, age, race)')
    parser.add_argument('--classifier_name', type=str, default='mlp', help='Name of the classifier (e.g., mlp, dt, rf)')
    parser.add_argument('--max_allowed_time', type=int, default=300, help='Maximum time allowed for the experiment')
    return parser.parse_args()


def get_experiment_params():
    args = parse_arguments()

    dataset_name = args.dataset_name
    sensitive_name = args.sensitive_name
    classifier_name = args.classifier_name

    config = get_data_config(dataset_name)
    sens_name = config.sens_name

    # Find the key corresponding to the sensitive name
    sensitive_param = None
    for key, value in sens_name.items():
        if value == sensitive_name:
            sensitive_param = key
            break

    if sensitive_param is None:
        available_options = ", ".join(sens_name.values())
        raise ValueError(f"Invalid sensitive name: {args.sensitive_name}. Available options are: {available_options}")

    return config, sensitive_name, sensitive_param, classifier_name, args.max_allowed_time


def get_data_dict():
    return {
        "census": census_data,
        "credit": credit_data,
        "bank": bank_data,
        "meps": meps_data,
        "compas": compas_data
    }


def get_config_dict():
    return {
        "census": census,
        "credit": credit,
        "bank": bank,
        "meps": meps,
        "compas": compas
    }


def validate_dataset_name(dataset_name, data_dict):
    if dataset_name not in data_dict:
        available_options = ", ".join(data_dict.keys())
        raise ValueError(f"Invalid dataset name: {dataset_name}. Available options are: {available_options}")


def get_data(dataset_name):
    data_dict = get_data_dict()
    validate_dataset_name(dataset_name, data_dict)
    return data_dict[dataset_name]


def get_data_config(dataset_name):
    config_dict = get_config_dict()
    validate_dataset_name(dataset_name, config_dict)
    return config_dict[dataset_name]

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, set):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return [numpy_to_python(item) for item in obj]
    return obj


def prepare_data(path, target_column):
    data = pd.read_csv(path)

    data_x = data.drop(columns=[target_column])
    data_y = data[target_column]
    feature_x = data_x.columns

    return data_x, data_y, feature_x


def save_results(results, path):
    # loop over the results, which is dictionary and create a dataframe based on that

    data = {'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]}

    for key, value in results.items():
        data[key] = [value]

    df = pd.DataFrame(data)

    if not os.path.isfile(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)


def cluster(dataset_name, X, cluster_num=4):
    model_path = f'../datasets/clusters/{dataset_name}.pkl'
    if os.path.exists(model_path):
        try:
            clf = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading pre-computed clusters: {e}")
            clf = None
    else:
        clf = None

    if clf is None:
        print(f"Computing clusters for {dataset_name}")
        clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(clf, model_path)
            print(f"Saved computed clusters to {model_path}")
        except Exception as e:
            print(f"Error saving computed clusters: {e}")

    return clf


def convert_to_np(data):
    inputs_array = np.array(list(data))
    if len(inputs_array.shape) > 2:
        inputs_array = inputs_array.squeeze(axis=1)
    return inputs_array


def safe_concatenate(arrays, axis=0):
    # Filter out empty arrays
    non_empty = [arr for arr in arrays if arr.size > 0]

    if not non_empty:
        # If all arrays are empty, return an empty array
        return np.array([])

    if len(non_empty) == 1:
        # If only one non-empty array, return it
        return non_empty[0]

    # Find the shape of the first non-empty array
    target_shape = list(non_empty[0].shape)
    target_shape[axis] = -1  # Allow any size along the concatenation axis

    # Reshape arrays to match the target shape
    reshaped = []
    for arr in non_empty:
        if arr.shape != tuple(target_shape):
            # Reshape only if necessary
            new_shape = list(arr.shape)
            new_shape[1:] = target_shape[1:]  # Match all dimensions except the first
            reshaped.append(np.reshape(arr, new_shape))
        else:
            reshaped.append(arr)

    # Concatenate the reshaped arrays
    return np.concatenate(reshaped, axis=axis)


def save_results_np(approach_name, dataset_name, sensitive_name, classifier_name, disc_inputs_array,
                    cumulative_efficiency_array):
    data_dir = os.path.join('results', 'data', approach_name, dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    disc_inputs_filename = os.path.join(data_dir,
                                        f'{classifier_name}_{sensitive_name}_disc_inputs.npy')
    cumulative_efficiency_filename = os.path.join(data_dir,
                                                  f'{classifier_name}_{sensitive_name}_cumulative_efficiency.npy')

    np.save(disc_inputs_filename, disc_inputs_array)
    np.save(cumulative_efficiency_filename, cumulative_efficiency_array)


def load_results_from_np_file(approach_name, dataset_name, sensitive_name, classifier_name):
    data_dir = os.path.join('results', 'data', approach_name, dataset_name)

    disc_indices_filename = os.path.join(data_dir,
                                         f'{classifier_name}_{sensitive_name}_disc_inputs.npy')
    cumulative_efficiency_filename = os.path.join(data_dir,
                                                  f'{classifier_name}_{sensitive_name}_cumulative_efficiency.npy')

    disc_inputs = np.load(disc_indices_filename)
    cumulative_efficiency = np.load(cumulative_efficiency_filename)

    return disc_inputs, cumulative_efficiency


def generate_report(
        approach_name,
        dataset_name,
        classifier_name,
        sensitive_name,
        samples,
        tot_inputs,
        disc_inputs,
        elapsed_time=0,
        save_report=True
):

    results = {
        'approach_name': approach_name,
        'dataset_name': dataset_name,
        'classifier_name': classifier_name,
        'sensitive_name': sensitive_name,
    }

    disc_rate = round((len(disc_inputs) / len(tot_inputs)) * 100, 4) if len(tot_inputs) > 0 else 0

    print(f'Total inputs: {len(tot_inputs)}')
    print(f'Total disc inputs: {len(disc_inputs)}')
    print(f'Disc rate: {disc_rate}')
    print(f'Elapsed time: {elapsed_time}')
    print('')

    results['samples'] = len(samples)
    results['tot_inputs'] = len(tot_inputs)
    results['disc_inputs'] = len(disc_inputs)
    results['disc_rate'] = disc_rate

    results['elapsed_time'] = elapsed_time
    results['egs'] = len(disc_inputs) / elapsed_time if elapsed_time > 0 else 0

    if save_report:
        data_dir = os.path.join('results')
        os.makedirs(data_dir, exist_ok=True)
        exp_path = f"{data_dir}/{approach_name}_experiment_results.csv"

        save_results(results, exp_path)
