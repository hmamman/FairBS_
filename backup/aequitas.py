import os

import numpy as np
import joblib
import time
import sys
import random

import pandas as pd
from scipy.optimize import basinhopping

# Get the absolute path to the directory where expga.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../../"))

from Genetic_Algorithm import GA
from utils.helpers import get_experiment_params, generate_report, get_data


class Aequitas:
    def __init__(self, classifier_name, config, sensitive_param, max_time_allowed=300, threshold=0):
        self.start_time = time.time()
        self.config = config
        self.init_prob = 0.5
        self.params = config.params
        self.direction_probability = [self.init_prob] * self.params
        self.direction_probability_change_size = 0.001

        self.param_probability = [1.0 / self.params] * self.params
        self.param_probability_change_size = 0.001

        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []

        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []

        self.tot_inputs = set()

        self.global_iteration_limit = 1000
        self.local_iteration_limit = 1000
        self.classifier_name = classifier_name
        self.classifier_path = f'models/{self.config.dataset_name}/{self.classifier_name}_standard_unfair.pkl'
        self.model = joblib.load(self.classifier_path)

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.tracking_interval = 100

        self.sensitive_param = sensitive_param
        self.threshold = threshold
        self.input_bounds = config.input_bounds
        self.perturbation_unit = 1
        self.max_time_allowed = max_time_allowed

        self.non_seed_inputs = set()
        self.non_seed_tot_inputs = set()
        self.non_seed_disc_inputs = set()
        self.non_seed_disc_inputs_list = []
        self.non_seed_cumulative_efficiency = []
        self.non_seed_time_to_1000_disc = -1

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

        # Loops through all values of the sensitive parameter
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

        # Returns early if input is already in the global discriminatory inputs set
        if tuple(map(tuple, inp0)) in self.global_disc_inputs:
            return 0

        out0 = self.model.predict(inp0)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.model.predict(inp1.reshape(1, -1))

                if abs(out1 - out0) > self.threshold:
                    self.global_disc_inputs.add(tuple(map(tuple, inp0)))  # add the entire input, including original y
                    self.global_disc_inputs_list.append(inp0.tolist()[0])

                    return abs(out0 - out1)

        self.non_seed_tot_inputs.add(tuple(map(tuple, inp0)))

        return 0

    def evaluate_local(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))
        # Returns early if input is already in the local discriminatory inputs set
        if (tuple(map(tuple, inp0)) in self.global_disc_inputs) or tuple(map(tuple, inp0)) in self.local_disc_inputs:
            return 0

        out0 = self.model.predict(inp0)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.model.predict(inp1.reshape(1, -1))

                if abs(out1 - out0) > self.threshold:
                    self.local_disc_inputs.add(tuple(map(tuple, inp0)))  # add the entire input, including original y
                    self.local_disc_inputs_list.append(inp0.tolist()[0])

                    return abs(out0 - out1)
        return 0

    def evaluate_fairbs(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)

        self.non_seed_tot_inputs.add(tuple(map(tuple, inp0)))
        self.total_generated += 1
        # Returns early if input is already in the local discriminatory inputs set
        if tuple(map(tuple, inp0)) in self.non_seed_disc_inputs:
            return 0

        out0 = self.model.predict(inp0)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.model.predict(inp1.reshape(1, -1))

                if abs(out1 - out0) > self.threshold:
                    self.non_seed_disc_inputs.add(tuple(map(tuple, inp0)))  # add the entire input, including original y
                    self.non_seed_disc_inputs_list.append(inp0.tolist()[0])

                    return 1
        return 0

    def run(self, max_local=1000, max_global=1000, max_allowed_time=300):
        elapsed_time = time.time() - self.start_time

        self.global_iteration_limit = max_global
        self.local_iteration_limit = max_local

        print("Search started")
        starting_time = time.time()

        minimizer = {"method": "L-BFGS-B"}

        global_discovery = self.GlobalDiscovery(self)
        local_perturbation = self.LocalPerturbation(self)

        basinhopping(self.evaluate_global, self.config.initial_input, stepsize=1.0, take_step=global_discovery,
                     minimizer_kwargs=minimizer, niter=self.global_iteration_limit)

        print(f'Total global generation: {len(self.global_disc_inputs_list)}')

        for inp in self.global_disc_inputs_list:
            basinhopping(self.evaluate_local, inp, stepsize=1.0, take_step=local_perturbation,
                         minimizer_kwargs=minimizer, niter=self.local_iteration_limit)

            elapsed_time = time.time() - self.start_time

            if elapsed_time >= max_allowed_time:
                break

        print(f'Total local generation: {len(self.local_disc_inputs_list)}')

        disc_inputs = self.local_disc_inputs_list
        disc_inputs.extend(self.global_disc_inputs_list)

        print(f'Aequitas total inputs: {len(self.tot_inputs)}')
        print(f'Aequitas total disc inputs: {len(disc_inputs)}')
        print(f'Aequitas disc %: {(len(disc_inputs) / len(self.tot_inputs)) * 100}')
        print(f'Aequitas speed %: {len(disc_inputs) / elapsed_time if elapsed_time > 0 else 0}')
        print(f'Aequitas elapsed time: {elapsed_time:.2f} seconds')

        print("")
        # print("Start FairBS")
        #
        # self.start_time = time.time()
        #
        # print(f'Total non seeds: {len(self.non_seed_inputs)}')
        #
        # data = get_data(self.config.dataset_name)
        # X, y, input_shape, nb_classes = data()
        #
        # Y = np.argmax(y, axis=1)
        #
        # population = [list(x[0]) for x in list(self.non_seed_inputs)[:200]]
        #
        # ga = GA(
        #     population=population,
        #     DNA_SIZE=len(self.config.input_bounds),
        #     bound=self.config.input_bounds,
        #     fitness_func=self.evaluate_fairbs,
        #     model=self.model,
        #     X=X,
        #     Y=Y
        # )
        #
        # for _ in range(max_local):
        #     ga.evolve()
        #     end = time.time() - self.start_time
        #     if end >= max_allowed_time:
        #         break
        #
        # elapsed_time = time.time() - self.start_time
        #
        # print(f'FairBS total inputs: {len(self.non_seed_tot_inputs)}')
        # print(f'FairBS total disc inputs: {len(self.non_seed_disc_inputs)}')
        # print(f'FairBS disc %: {(len(self.non_seed_disc_inputs) / len(self.non_seed_tot_inputs)) * 100}')
        # print(f'FairBS speed %: {len(self.non_seed_disc_inputs) / elapsed_time if elapsed_time > 0 else 0}')
        # print(f'FairBS elapsed time: {elapsed_time:.2f} seconds')

        # generate_report(
        #     approach_name='FIGGA',
        #     dataset_name=self.config.dataset_name,
        #     classifier_name=self.classifier_name,
        #     sensitive_name=self.config.sens_name[self.sensitive_param],
        #     tot_inputs=self.tot_inputs,
        #     disc_inputs=self.disc_inputs_list,
        #     total_generated_inputs=self.total_generated,
        #     elapsed_time=elapsed_time,
        #     time_to_1000_disc=self.time_to_1000_disc,
        #     cumulative_efficiency=self.cumulative_efficiency,
        #     save_report=True
        # )

if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

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

    aequitas.run(max_allowed_time=max_allowed_time)
