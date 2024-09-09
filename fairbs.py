import random

import joblib
import time
import numpy as np
import pandas as pd
import dice_ml
import json
from utils import helpers


class FairBS:
    def __init__(self, classifier_name, config, sensitive_param, max_time_allowed=300, threshold=0):
        self.config = config
        self.classifier_name = classifier_name
        self.classifier_path = f'models/{self.config.dataset_name}/{self.classifier_name}_classifier.pkl'
        self.model = joblib.load(self.classifier_path)
        self.sensitive_param = sensitive_param
        self.threshold = threshold
        self.input_bounds = config.input_bounds
        self.max_time_allowed = max_time_allowed
        self.no_seed_inputs = set()
        self.tot_inputs = set()
        self.disc_cf_inputs = set()
        self.disc_cf_inputs_list = []
        self.exp = self.get_explainer()

    def get_explainer(self):
        features = {f'{f}': r for f, r in zip(self.config.feature_name, self.config.input_bounds)}
        d = dice_ml.Data(features=features, continuous_features=[], outcome_name='y')
        backend = 'sklearn'
        m = dice_ml.Model(model=self.model, backend=backend)
        method = "random"
        return dice_ml.Dice(d, m, method=method)

    def generate_counterfactuals(self, inp, cf_limit, desired_class='opposite'):
        # query_instance = self.inp_to_df(inp)
        # features_to_vary = [self.config.feature_name[i] for i in range(len(self.config.feature_name)) if
        #                     i is not self.sensitive_param - 1]
        #
        # dice_exp = self.exp.generate_counterfactuals(
        #     query_instance,
        #     total_CFs=cf_limit,
        #     features_to_vary=features_to_vary,
        #     desired_class=desired_class)
        #
        # return json.loads(dice_exp.to_json())['cfs_list'][0]
        try:
            query_instance = self.inp_to_df(inp)
            features_to_vary = [self.config.feature_name[i] for i in range(len(self.config.feature_name)) if
                                i is not self.sensitive_param - 1]

            dice_exp = self.exp.generate_counterfactuals(
                query_instance,
                total_CFs=cf_limit,
                features_to_vary=features_to_vary,
                desired_class=desired_class)

            return json.loads(dice_exp.to_json())['cfs_list'][0]
        except:
            pass
        return []

    def inp_to_df(self, inp):
        return pd.DataFrame(inp, columns=self.config.feature_name)

    def make_prediction(self, inp):
        return self.model.predict(self.inp_to_df(inp))

    def non_seed_discovery(self, non_seed_limit=1000):
        no_seed_inputs = []
        random.seed(time.time())
        inputs = [
            np.array([random.randint(low, high) for [low, high] in self.input_bounds]) for _ in range(non_seed_limit)
        ]

        for inp in inputs:
            if not self.check_is_seed(inp):
                no_seed_inputs.append(inp)

        return no_seed_inputs

    def counterfactual_discovery(self, inp, cf_limit=1000):
        inp0 = inp.reshape(1, -1)
        if tuple(map(tuple, inp0)) not in self.tot_inputs:

            cfs = self.generate_counterfactuals(inp0, cf_limit=cf_limit, desired_class='opposite')
            for x in cfs:
                self.evaluate_counterfactual(x[:-1])

    def check_is_seed(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)

        out0 = self.make_prediction(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.make_prediction(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.disc_cf_inputs.add(tuple(map(tuple, inp0)))
                    self.disc_cf_inputs_list.append(inp0.tolist()[0])
                    return True
        return False

    def evaluate_counterfactual(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]
        inp0 = inp0.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))

        if tuple(map(tuple, inp0)) in self.disc_cf_inputs:
            return 0

        out0 = self.make_prediction(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.make_prediction(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.disc_cf_inputs.add(tuple(map(tuple, inp0)))
                    self.disc_cf_inputs_list.append(inp0.tolist()[0])
                    return 1
        return 0

    def run_fairbs(self, non_seed_limit=1000, cf_limit=1000, max_allowed_time=300):
        start_time = time.time()
        no_seed_inputs = self.non_seed_discovery(non_seed_limit=non_seed_limit)

        for inp in no_seed_inputs:
            self.counterfactual_discovery(inp=inp, cf_limit=cf_limit)

            end = time.time() - start_time
            if end >= max_allowed_time:
                break

        elapsed_time = time.time() - start_time

        helpers.generate_report(
            approach_name='TestFairBS',
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            samples=no_seed_inputs,
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_cf_inputs_list,
            elapsed_time=elapsed_time,
            save_report=True
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = helpers.get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    experiment = FairBS(
        config=config,
        classifier_name=classifier_name,
        sensitive_param=sensitive_param
    )

    experiment.run_fairbs(max_allowed_time=max_allowed_time)
