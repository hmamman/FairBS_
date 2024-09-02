import os
import sys

import time

import joblib
import numpy as np

# Get the absolute path to the directory where figga.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../../"))

from Figga_Genetic_Algorithm import GA
from utils.helpers import get_experiment_params, generate_report, get_data


class FIGGA:
    def __init__(self, config, classifier_name, sensitive_param, population_size=200, threshold=0):
        self.config = config
        self.threshold = threshold
        self.binary_threshold = 0
        self.distance_threshold = 0.2
        self.population_size = population_size

        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.input_bounds = np.array(self.config.input_bounds)
        self.sensitive_param = sensitive_param
        self.classifier_name = classifier_name
        self.classifier_path = f'models/{self.config.dataset_name}/{self.classifier_name}_standard_unfair.pkl'
        self.model = joblib.load(self.classifier_path)

        self.start_time = time.time()
        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.tracking_interval = 100

    def evaluate_disc(self, inp):
        inp0 = np.array([int(x) for x in inp])
        inp1 = np.array([int(x) for x in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)
        inp1 = inp1.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))

        self.total_generated += 1
        self.update_cumulative_efficiency()

        if tuple(map(tuple, inp0)) not in self.disc_inputs:
            out0 = self.model.predict(inp0)

            sensitive_values = range(self.config.input_bounds[self.sensitive_param - 1][0],
                                     self.config.input_bounds[self.sensitive_param - 1][1] + 1)

            for i in sensitive_values:
                if i != original_sensitive_value:
                    inp1[0][self.sensitive_param - 1] = i
                    out1 = self.model.predict(inp1)

                    if abs(out0 - out1) > self.threshold:
                        self.disc_inputs.add(tuple(map(tuple, inp0)))
                        self.disc_inputs_list.append(inp0.tolist()[0])

                        self.set_time_to_1000_disc()

                        return 1

        return 0

    def evaluate_hybrid_disc(self, inp):
        binary_disc = False
        max_distance = 0
        discriminatory_pairs = []

        inp0 = np.array([int(x) for x in inp])
        inp1 = np.array([int(x) for x in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)
        inp1 = inp1.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))

        self.total_generated += 1
        self.update_cumulative_efficiency()

        if tuple(map(tuple, inp0)) not in self.disc_inputs:

            sensitive_values = range(self.config.input_bounds[self.sensitive_param - 1][0],
                                     self.config.input_bounds[self.sensitive_param - 1][1] + 1)

            for i in sensitive_values:
                if i != original_sensitive_value:
                    inp1[0][self.sensitive_param - 1] = i
                    if self.binary_discrimination_check(inp0, inp1):
                        binary_disc = True
                        discriminatory_pairs.append(inp0.copy())
                        break

                    distance = self.calculate_distance(inp0, inp1)
                    if round(distance, 1) > self.distance_threshold:
                        max_distance = distance
                        if distance > self.distance_threshold:
                            print(f"disc based on distance: %f", distance)
                            break

        if binary_disc or max_distance > self.distance_threshold:
            self.disc_inputs.add(tuple(map(tuple, inp0)))
            self.disc_inputs_list.append(inp0.tolist()[0])

            self.set_time_to_1000_disc()

        return 1 if binary_disc else max_distance

    def binary_discrimination_check(self, input1, input2):
        pred1 = self.model.predict(input1.reshape(1, -1))
        pred2 = self.model.predict(input2.reshape(1, -1))
        return abs(pred1 - pred2) > self.binary_threshold

    def calculate_distance(self, input1, input2):
        prob1 = self.model.predict_proba(input1.reshape(1, -1))
        prob2 = self.model.predict_proba(input2.reshape(1, -1))
        return np.linalg.norm(prob1 - prob2)

    def update_cumulative_efficiency(self):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)

        if total_inputs == 1 or (total_inputs % self.tracking_interval == 0 and
                                 (not self.cumulative_efficiency or
                                  self.cumulative_efficiency[-1][0] != total_inputs)):
            total_disc_inputs = len(self.disc_inputs)
            self.cumulative_efficiency.append([total_inputs, total_disc_inputs])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)

        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_evolution=1000, max_allowed_time=300):

        data = get_data(self.config.dataset_name)
        X, y, input_shape, nb_classes = data()

        Y = np.argmax(y, axis=1)

        ga = GA(
            pop_size=self.population_size,
            DNA_SIZE=len(self.config.input_bounds),
            bound=self.config.input_bounds,
            fitness_func=self.evaluate_hybrid_disc,
            model=self.model,
            X=X,
            Y=Y
        )

        for _ in range(max_evolution):
            ga.evolve()
            end = time.time() - self.start_time
            if end >= max_allowed_time:
                break

        elapsed_time = time.time() - self.start_time

        generate_report(
            approach_name='FIGGA',
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs_list,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            save_report=True
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    # print(f'Approach name: ExpGA')
    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    figga = FIGGA(
        config=config,
        classifier_name=classifier_name,
        sensitive_param=sensitive_param
    )

    figga.run(max_allowed_time=max_allowed_time)
