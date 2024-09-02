import argparse

from aequitas_fairbs import AequitasFairBS
from utils.ml_classifiers import CLASSIFIERS
from utils.helpers import get_config_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--approach_name', type=str, default='both', required=False, help='The name of fairness testing approach to run')
    parser.add_argument('--max_allowed_time', type=int, default=300, help='Maximum time allowed for the experiment')
    parser.add_argument('--max_iteration', type=int, default=1, help='Maximum experiment iterations')
    return parser.parse_args()


args = parse_arguments()
approaches = ['aequitas', 'fairbs', 'both']

if args.approach_name not in approaches:
    available_options = ", ".join(approaches)
    raise ValueError(f"Invalid sensitive name: {args.approach_name}. Available options are: {approaches}")

for config in get_config_dict().values():

    for classifier_name in CLASSIFIERS.keys():
        for sensitive_param in config.sens_name:

            for _ in range(args.max_iteration):
                print(f'Approach name: {args.approach_name}')
                print(f'Dataset: {config.dataset_name}')
                print(f'Classifier: {classifier_name}')
                print(f'Sensitive name: {config.sens_name[sensitive_param]}')

                experiment = AequitasFairBS(
                    config=config,
                    classifier_name=classifier_name,
                    sensitive_param=sensitive_param
                )

                experiment.run_global(max_allowed_time=args.max_allowed_time)

                if args.approach_name == 'aequitas':
                    experiment.run_aequitas(max_allowed_time=args.max_allowed_time)
                elif args.approach_name == 'fairbs':
                    experiment.run_fairbs(max_allowed_time=args.max_allowed_time)
                else:
                    experiment.run_aequitas_fairbs(max_allowed_time=args.max_allowed_time)
