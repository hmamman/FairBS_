import os
import numpy as np
import joblib
import time
import random
import json
import dice_ml
import pandas as pd
from scipy.optimize import basinhopping
from utils import helpers


class Aequitas:
    def __init__(self, classifier_name, config, sensitive_param, max_time_allowed=300, threshold=0):
        """
        Initialize the Aequitas fairness testing framework.

        Args:
            classifier_name (str): Name of the classifier to be tested.
            config (object): Configuration object containing dataset and model parameters.
            sensitive_param (int): Index of the sensitive parameter in the input vector.
            max_time_allowed (int, optional): Maximum time allowed for testing in seconds. Defaults to 300.
            threshold (float, optional): Threshold for considering an output difference as discriminatory. Defaults to 0.
        """
        self.start_time = time.time()
        self.config = config
        self.init_prob = 0.5
        self.params = config.params
        self.direction_probability = [self.init_prob] * self.params
        self.direction_probability_change_size = 0.001

        self.param_probability = [1.0 / self.params] * self.params
        self.param_probability_change_size = 0.001

        self.themis_disc_inputs = set()
        self.themis_disc_inputs_list = []

        self.aequitas_disc_inputs = set()
        self.aequitas_disc_inputs_list = []

        self.tot_inputs = set()

        self.themis_iteration_limit = 1000
        self.aequitas_iteration_limit = 1000
        self.classifier_name = classifier_name
        self.classifier_path = f'models/{self.config.dataset_name}/{self.classifier_name}_standard_unfair.pkl'
        self.model = joblib.load(self.classifier_path)

        self.sensitive_param = sensitive_param
        self.threshold = threshold
        self.input_bounds = config.input_bounds
        self.perturbation_unit = 1
        self.max_time_allowed = max_time_allowed

        self.non_seed_inputs = set()
        self.fairbs_tot_inputs = set()
        self.fairbs_disc_inputs = set()
        self.fairbs_disc_inputs_list = []
        self.exp = self.get_explainer()

    def normalize_probability(self):
        """Normalize the parameter probability distribution."""
        probability_sum = sum(self.param_probability)
        self.param_probability = [float(prob) / float(probability_sum) for prob in self.param_probability]

    class AequitasPerturbation:
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
                self.parent.normalize_probability()
            else:
                self.parent.param_probability[param_choice] = max(
                    self.parent.param_probability[param_choice] - self.parent.param_probability_change_size, 0)
                self.parent.normalize_probability()

            return x

    class Themis:
        def __init__(self, parent, stepsize=1):
            self.parent = parent
            self.stepsize = stepsize

        def __call__(self, x):
            for i in range(self.parent.params):
                random.seed(time.time())
                x[i] = random.randint(self.parent.input_bounds[i][0], self.parent.input_bounds[i][1])

            x[self.parent.sensitive_param - 1] = 0
            return x

    def evaluate(self, inp, algorithm):
        """
        Evaluate an input using the specified algorithm to detect discriminatory instances.

        Args:
            inp (numpy.array): Input vector to be evaluated.
            algorithm (str): The algorithm to use ('themis', 'aequitas', or 'fairbs').

        Returns:
            float or int: The evaluation result (difference in outputs or 1/0 for discriminatory/non-discriminatory).
        """
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]

        frozen_inp = frozenset(map(tuple, inp0.reshape(1, -1)))

        if algorithm == 'themis':
            self.tot_inputs.add(frozen_inp)
            if frozen_inp in self.themis_disc_inputs:
                return 0
            disc_inputs = self.themis_disc_inputs
            disc_inputs_list = self.themis_disc_inputs_list
        elif algorithm == 'aequitas':
            self.tot_inputs.add(frozen_inp)
            if frozen_inp in self.themis_disc_inputs or frozen_inp in self.aequitas_disc_inputs:
                return 0
            disc_inputs = self.aequitas_disc_inputs
            disc_inputs_list = self.aequitas_disc_inputs_list
        elif algorithm == 'fairbs':
            self.fairbs_tot_inputs.add(frozen_inp)
            if frozen_inp in self.fairbs_disc_inputs:
                return 0
            disc_inputs = self.fairbs_disc_inputs
            disc_inputs_list = self.fairbs_disc_inputs_list
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        out0 = self.model.predict(self.input_to_dataframe(inp0))

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i

                out1 = self.model.predict(self.input_to_dataframe(inp1))

                if abs(out1 - out0) > self.threshold:
                    disc_inputs.add(frozen_inp)
                    disc_inputs_list.append(inp0.reshape(1, -1) if algorithm == 'fairbs' else inp0)
                    return 1 if algorithm == 'fairbs' else abs(out0 - out1)

        if algorithm == 'themis':
            self.non_seed_inputs.add(frozen_inp)

        return 0

    # Replace individual evaluation methods with calls to the new evaluate method
    def evaluate_themis(self, inp):
        return self.evaluate(inp, 'themis')

    def evaluate_aequitas(self, inp):
        return self.evaluate(inp, 'aequitas')

    def evaluate_fairbs(self, inp):
        return self.evaluate(inp, 'fairbs')

    def get_explainer(self):
        """
        Get the DiCE explainer object for generating counterfactuals.

        Returns:
            dice_ml.Dice: DiCE explainer object.
        """
        features = {f'{f}': r for f, r in zip(self.config.feature_name, self.config.input_bounds)}
        d = dice_ml.Data(features=features,
                         continuous_features=[],
                         outcome_name='y')
        backend = 'sklearn'
        m = dice_ml.Model(model=self.model, backend=backend)
        method = "random"
        exp = dice_ml.Dice(d, m, method=method)
        return exp

    def generate_counterfactuals(self, inp, desired_class='opposite'):
        """
        Generate counterfactual examples for a given input.

        Args:
            inp (numpy.array): Input vector for which to generate counterfactuals.
            desired_class (str, optional): Desired output class for counterfactuals. Defaults to 'opposite'.

        Returns:
            list: List of counterfactual examples.
        """
        try:
            query_instance = self.input_to_dataframe(inp)
            total_CFs = self.aequitas_iteration_limit
            features_to_vary = [self.config.feature_name[i] for i in range(len(self.config.feature_name)) if
                                i is not self.sensitive_param - 1]

            dice_exp = self.exp.generate_counterfactuals(
                query_instance,
                total_CFs=total_CFs,
                features_to_vary=features_to_vary,
                desired_class=desired_class)

            return json.loads(dice_exp.to_json())['cfs_list'][0]
        except Exception as e:
            print(f"Error generating counterfactuals: {str(e)}")
            return []

    def input_to_dataframe(self, inp):
        """
        Convert input array to pandas DataFrame.

        Args:
            inp (numpy.array): Input vector.

        Returns:
            pandas.DataFrame: DataFrame representation of the input.
        """
        return pd.DataFrame([inp], columns=self.config.feature_name)

    def make_prediction(self, inp):
        """
        Make a prediction using the loaded model.

        Args:
            inp (numpy.array): Input vector.

        Returns:
            The model's prediction for the given input.
        """
        return self.model.predict(self.input_to_dataframe(inp))

    def run(self, max_aequitas=1000, max_themis=1000, max_allowed_time=300):
        """
        Run the Aequitas fairness testing framework.

        Args:
            max_aequitas (int): Maximum number of iterations for Aequitas algorithm.
            max_themis (int): Maximum number of iterations for Themis algorithm.
            max_allowed_time (int): Maximum allowed runtime in seconds.
        """
        self.start_time = time.time()
        self.themis_iteration_limit = max_themis
        self.aequitas_iteration_limit = max_aequitas

        # Common configuration for basin-hopping
        minimizer = {"method": "L-BFGS-B"}

        # Run Themis algorithm
        self._run_themis(minimizer)

        # Run Aequitas algorithm
        self._run_aequitas(minimizer, max_allowed_time)

        # Run FairBS algorithm
        self._run_fairbs(max_allowed_time)

    def _run_themis(self, minimizer):
        """Run the Themis algorithm and generate report."""
        print("Themis search started")
        themis = self.Themis(self)
        basinhopping(self.evaluate, self.config.initial_input, stepsize=1.0, take_step=themis,
                     minimizer_kwargs=minimizer, niter=self.themis_iteration_limit,
                     args=('themis',))  # Pass 'themis' as argument to evaluate method

        elapsed_time = time.time() - self.start_time
        print(f'Total Themis generation: {len(self.themis_disc_inputs_list)}')

        self._generate_report('THEMIS', self.themis_disc_inputs_list, elapsed_time)

    def _run_aequitas(self, minimizer, max_allowed_time):
        """Run the Aequitas algorithm and generate report."""
        print("\nAequitas search started")
        aequitas_perturbation = self.AequitasPerturbation(self)

        for inp in self.themis_disc_inputs_list:
            basinhopping(self.evaluate, inp, stepsize=1.0, take_step=aequitas_perturbation,
                         minimizer_kwargs=minimizer, niter=self.aequitas_iteration_limit,
                         args=('aequitas',))  # Pass 'aequitas' as argument to evaluate method

            if time.time() - self.start_time >= max_allowed_time:
                print("Max allowed time reached. Stopping Aequitas search.")
                break

        elapsed_time = time.time() - self.start_time
        print(f'Total Aequitas generation: {len(self.aequitas_disc_inputs_list)}')

        disc_inputs = self.aequitas_disc_inputs_list + self.themis_disc_inputs_list
        self._generate_report('AEQUITAS', disc_inputs, elapsed_time)

    def _run_fairbs(self, max_allowed_time):
        """Run the FairBS algorithm and generate report."""
        print("\nFairBS search started")
        self.start_time = time.time()
        samples = [list(x) for x in self.non_seed_inputs]  # Remove [0] indexing as it's not needed

        for inp in samples:
            counterfactuals = self.generate_counterfactuals(inp)  # Use the renamed method
            for cf in counterfactuals:
                self.evaluate(cf[:-1], 'fairbs')  # Use the new evaluate method

            if time.time() - self.start_time >= max_allowed_time:
                print("Max allowed time reached. Stopping FairBS search.")
                break

        elapsed_time = time.time() - self.start_time
        print(f'Total FairBS generation: {len(self.fairbs_disc_inputs_list)}')

        self._generate_report('FairBS', self.fairbs_disc_inputs_list, elapsed_time)

    def _generate_report(self, approach_name, disc_inputs, elapsed_time):
        """Generate and save a report for the given approach."""
        helpers.generate_report(
            approach_name=approach_name,
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs if approach_name != 'FairBS' else self.fairbs_tot_inputs,
            disc_inputs=disc_inputs,
            elapsed_time=elapsed_time,
            save_report=True
        )

if __name__ == '__main__':
    # Load experiment parameters
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = helpers.get_experiment_params()

    # Print experiment details
    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive attribute: {sensitive_name}')
    print(f'Max allowed time: {max_allowed_time} seconds')
    print('')

    # Initialize and run Aequitas
    aequitas = Aequitas(
        config=config,
        classifier_name=classifier_name,
        sensitive_param=sensitive_param
    )

    aequitas.run(max_allowed_time=max_allowed_time)

    print("\nExperiment completed.")