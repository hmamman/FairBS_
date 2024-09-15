import numpy as np
import joblib
import time
import random
import pandas as pd
from scipy.optimize import basinhopping
from utils import helpers


class Aequitas:
    def __init__(self, classifier_name, config, sensitive_param, max_time_allowed=300, threshold=0):
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
        out0 = self.model.predict(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.model.predict(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    return True
        return False

    def evaluate_global(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]
        inp0 = inp0.reshape(1, -1)
        self.tot_inputs.add(tuple(map(tuple, inp0)))

        if tuple(map(tuple, inp0)) in self.local_disc_inputs:
            return 0

        out0 = self.model.predict(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.model.predict(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.global_disc_inputs.add(tuple(map(tuple, inp0)))
                    self.global_disc_inputs_list.append(inp0.tolist()[0])
                    return abs(out0 - out1)
        return 0

    def evaluate_local(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]
        inp0 = inp0.reshape(1, -1)
        self.tot_inputs.add(tuple(map(tuple, inp0)))

        if (tuple(map(tuple, inp0)) in self.local_disc_inputs) or tuple(map(tuple, inp0)) in self.global_disc_inputs:
            return 0

        out0 = self.model.predict(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.model.predict(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.local_disc_inputs.add(tuple(map(tuple, inp0)))
                    self.local_disc_inputs_list.append(inp0.tolist()[0])
                    return abs(out0 - out1)
        return 0

    def run_aequitas(self, max_global=1000, max_local=1000, max_allowed_time=300):
        start_time = time.time()
        self.global_iteration_limit = max_global
        self.local_iteration_limit = max_local

        minimizer = {"method": "L-BFGS-B"}

        global_discovery = self.GlobalDiscovery(self)
        local_perturbation = self.LocalPerturbation(self)

        print("Search started")

        basinhopping(self.evaluate_global, self.config.initial_input, stepsize=1.0, take_step=global_discovery,
                     minimizer_kwargs=minimizer, niter=self.global_iteration_limit)
        print(f'Total global generation: {len(self.global_disc_inputs)}')

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


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = helpers.get_experiment_params()

    # print(f'Approach name: ExpGA')
    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    aequitas = Aequitas(
        config=config,
        classifier_name=classifier_name,
        sensitive_param=sensitive_param
    )

    aequitas.run_aequitas(max_allowed_time=max_allowed_time)
