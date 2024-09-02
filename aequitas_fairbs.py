import os

import numpy as np
import joblib
import time
import sys
import random
import json

import dice_ml

import pandas as pd
from scipy.optimize import basinhopping

from utils import helpers


class AequitasFairBS:
    def __init__(self, classifier_name, config, sensitive_param, max_time_allowed=300, threshold=0):
        self.start_time = time.time()
        self.config = config
        self.init_prob = 0.5
        self.params = config.params
        self.direction_probability = [self.init_prob] * self.params
        self.direction_probability_change_size = 0.001

        self.param_probability = [1.0 / self.params] * self.params
        self.param_probability_change_size = 0.001

        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []

        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []

        self.tot_inputs = set()

        self.local_iteration_limit = 1000
        self.global_iteration_limit = 1000
        self.classifier_name = classifier_name
        self.classifier_path = f'models/{self.config.dataset_name}/{self.classifier_name}_classifier.pkl'
        self.model = joblib.load(self.classifier_path)

        self.sensitive_param = sensitive_param
        self.threshold = threshold
        self.input_bounds = config.input_bounds
        self.perturbation_unit = 1
        self.max_time_allowed = max_time_allowed

        self.no_seed_inputs = set()
        self.fairbs_tot_inputs = set()
        self.fairbs_disc_inputs = set()
        self.fairbs_disc_inputs_list = []
        self.exp = self.get_explainer()

    def normalise_probability(self):
        probability_sum = sum(self.param_probability)
        self.param_probability = [float(prob) / float(probability_sum) for prob in self.param_probability]

    class LocalPerturbation:
        def __init__(self, parent, stepsize=1):
            self.parent = parent
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            param_choice = np.random.choice(range(self.parent.params), p=self.parent.param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[self.parent.direction_probability[param_choice],
                                                        (1 - self.parent.direction_probability[param_choice])])

            if (x[param_choice] == self.parent.input_bounds[param_choice][0]) or (
                    x[param_choice] == self.parent.input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * self.parent.perturbation_unit)

            x[param_choice] = max(self.parent.input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(self.parent.input_bounds[param_choice][1], x[param_choice])

            ei = self.parent.evaluate_input(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                self.parent.direction_probability[param_choice] = min(
                    self.parent.direction_probability[param_choice] + (
                            self.parent.direction_probability_change_size * self.parent.perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                self.parent.direction_probability[param_choice] = max(
                    self.parent.direction_probability[param_choice] - (
                            self.parent.direction_probability_change_size * self.parent.perturbation_unit), 0)

            if ei:
                self.parent.param_probability[param_choice] = self.parent.param_probability[
                                                                  param_choice] + self.parent.param_probability_change_size
                self.parent.normalise_probability()
            else:
                self.parent.param_probability[param_choice] = max(
                    self.parent.param_probability[param_choice] - self.parent.param_probability_change_size, 0)
                self.parent.normalise_probability()

            return x

    class GlobalDiscovery:
        def __init__(self, parent, stepsize=1):
            self.parent = parent
            self.stepsize = stepsize

        def __call__(self, x):
            for i in range(self.parent.params):
                random.seed(time.time())
                x[i] = random.randint(self.parent.input_bounds[i][0], self.parent.input_bounds[i][1])

            x[self.parent.sensitive_param - 1] = 0
            return x

    def evaluate_input(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)
        out0 = self.make_prediction(inp0)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.make_prediction(inp1.reshape(1, -1))

                if abs(out1 - out0) > self.threshold:
                    return True
        return False

    def evaluate_global(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))

        # Returns early if input is already in the local discriminatory inputs set
        if tuple(map(tuple, inp0)) in self.local_disc_inputs:
            return 0

        out0 = self.make_prediction(inp0)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.make_prediction(inp1.reshape(1, -1))

                if abs(out1 - out0) > self.threshold:
                    self.global_disc_inputs.add(
                        tuple(map(tuple, inp0)))  # add the entire input, including original y
                    self.global_disc_inputs_list.append(inp0.tolist()[0])

                    return abs(out0 - out1)
        # Record the non-seed instances
        self.no_seed_inputs.add(tuple(map(tuple, inp0)))

        return 0

    def evaluate_local(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))
        # Returns early if input is already in the global discriminatory inputs set
        if (tuple(map(tuple, inp0)) in self.local_disc_inputs) or tuple(map(tuple, inp0)) in self.global_disc_inputs:
            return 0

        out0 = self.make_prediction(inp0)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):

            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.make_prediction(inp1.reshape(1, -1))

                if abs(out1 - out0) > self.threshold:
                    self.local_disc_inputs.add(tuple(map(tuple, inp0)))  # add the entire input, including original y
                    self.local_disc_inputs_list.append(inp0.tolist()[0])

                    return abs(out0 - out1)
        return 0

    def evaluate_cf(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)

        self.fairbs_tot_inputs.add(tuple(map(tuple, inp0)))
        # Returns early if input is already in the global discriminatory inputs set
        if tuple(map(tuple, inp0)) in self.fairbs_disc_inputs:
            return 0

        out0 = self.make_prediction(inp0)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):

            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.make_prediction(inp1.reshape(1, -1))

                if abs(out1 - out0) > self.threshold:
                    self.fairbs_disc_inputs.add(tuple(map(tuple, inp0)))  # add the entire input, including original y
                    self.fairbs_disc_inputs_list.append(inp0.tolist()[0])

                    return 1
        return 0

    def get_explainer(self):
        # Dataset for training an ML model
        features = {f'{f}': r for f, r in zip(self.config.feature_name, self.config.input_bounds)}
        d = dice_ml.Data(features=features,
                         continuous_features=[],
                         outcome_name='y')
        # Pre-trained ML model
        # provide the trained ML model to DiCE's model object
        backend = 'sklearn'
        m = dice_ml.Model(model=self.model, backend=backend)
        # DiCE explanation instance
        method = "random"
        exp = dice_ml.Dice(d, m, method=method)

        return exp

    def generate_cfs(self, inp, cf_limit, desired_class='opposite'):
        # Generate counterfactual examples
        try:
            query_instance = self.inp_to_df(inp)
            features_to_vary = [self.config.feature_name[i] for i in range(len(self.config.feature_name)) if
                                i is not self.sensitive_param - 1]

            dice_exp = self.exp.generate_counterfactuals(
                query_instance,
                total_CFs=cf_limit,
                features_to_vary=features_to_vary,
                desired_class=desired_class)
            # Visualize counterfactual explanation

            return json.loads(dice_exp.to_json())['cfs_list'][0]
        except:
            pass
        return []

    def inp_to_df(self, inp):
        return pd.DataFrame(inp, columns=self.config.feature_name)

    def make_prediction(self, inp):
        return self.model.predict(self.inp_to_df(inp))

    def run_global(self, max_global=1000, max_allowed_time=300):
        self.global_iteration_limit = max_global

        self.start_time = time.time()

        print("Search started")

        minimizer = {"method": "L-BFGS-B"}

        global_discovery = self.GlobalDiscovery(self)

        basinhopping(self.evaluate_global, self.config.initial_input, stepsize=1.0, take_step=global_discovery,
                     minimizer_kwargs=minimizer, niter=self.global_iteration_limit)

        print(f'Total global generation: {len(self.global_disc_inputs)}')

    def run_aequitas(self, max_global=1000, max_allowed_time=300):
        self.global_iteration_limit = max_global

        start_time = time.time()

        print("Search started")

        minimizer = {"method": "L-BFGS-B"}
        local_perturbation = self.LocalPerturbation(self)

        for inp in self.global_disc_inputs_list:
            basinhopping(self.evaluate_local, inp, stepsize=1.0, take_step=local_perturbation,
                         minimizer_kwargs=minimizer, niter=self.local_iteration_limit)

            end = time.time() - start_time

            if end >= max_allowed_time:
                break

        elapsed_time = time.time() - start_time

        print(f'Total local generation: {len(self.local_disc_inputs)}')

        disc_inputs = self.local_disc_inputs | self.global_disc_inputs

        helpers.generate_report(
            approach_name='AEQUITAS',
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            samples=self.global_disc_inputs,
            tot_inputs=self.tot_inputs,
            disc_inputs=disc_inputs,
            elapsed_time=elapsed_time,
            save_report=True
        )

    def run_fairbs(self, cf_limit=1000, max_allowed_time=300):
        start_time = time.time()

        samples = [list(x[0]) for x in self.no_seed_inputs]

        for inp in samples:
            cfs = self.generate_cfs([inp], cf_limit)
            for c in cfs:
                self.evaluate_cf(c[:-1])

            end = time.time() - start_time
            if end >= max_allowed_time:
                break

        elapsed_time = time.time() - start_time

        helpers.generate_report(
            approach_name='FairBS',
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            samples=self.no_seed_inputs,
            tot_inputs=self.fairbs_tot_inputs,
            disc_inputs=self.fairbs_disc_inputs_list,
            elapsed_time=elapsed_time,
            save_report=True
        )

    def run_aequitas_fairbs(self, max_allowed_time=300):
        self.run_aequitas(max_allowed_time=max_allowed_time)
        self.run_fairbs(max_allowed_time=max_allowed_time)

        elapsed_time = time.time() - self.start_time

        disc_inputs = self.local_disc_inputs | self.global_disc_inputs | self.fairbs_disc_inputs
        tot_inputs = self.tot_inputs | self.fairbs_tot_inputs

        helpers.generate_report(
            approach_name='AEQUITAS+FairBS',
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            samples=self.global_disc_inputs | self.no_seed_inputs,
            tot_inputs=tot_inputs,
            disc_inputs=disc_inputs,
            elapsed_time=elapsed_time,
            save_report=True
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = helpers.get_experiment_params()

    # print(f'Approach name: AEQUITAS FAIRBS')
    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    experiment = AequitasFairBS(
        config=config,
        classifier_name=classifier_name,
        sensitive_param=sensitive_param
    )

    experiment.run_global(max_allowed_time=max_allowed_time)
    experiment.run_aequitas_fairbs(max_allowed_time=max_allowed_time)
