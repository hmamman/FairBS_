# FairBS
Code for the paper "Beyond the Seeds: Fairness Testing via Counterfactual Analysis of Non-Seed Instances" by Hussaini Mamman, Shuib Basri, Abdullateef Oluwagbemiga Balogun, Abdul Rehman Gilal, Abdullahi Abubakar Imam, Ganesh Kumar and Luiz Fernando Capretz.

[//]: # (![Overview of ExpGA]&#40;./figures/FairBS Framework.png&#41;)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hmamman/FairBS.git
   cd FairBS
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


## Running Fairness Testing

The provided Python script is designed to run a fairness testing approach based on user-specified parameters. The script accepts several command-line arguments, which can be customized to control the dataset, classifier, sensitive parameter, and experiment duration.

### Command-Line Arguments

The script accepts the following arguments:

- `--dataset_name`: (string) Name of the dataset to use in the experiment. The default is `'census'`.
  - Example: `--dataset_name census`

- `--sensitive_name`: (string) Name of the sensitive attribute for fairness testing (e.g., `sex`, `age`, `race`). The default is `'age'`.
  - Example: `--sensitive_name sex`

- `--classifier_name`: (string) Name of the classifier to use (e.g., `mlp`, `dt`, `rf`). The default is `'mlp'`.
  - Example: `--classifier_name rf`

- `--max_allowed_time`: (integer) Maximum time in seconds for the experiment to run. The default is `300` seconds (5 minutes).
  - Example: `--max_allowed_time 600`

### Example Usage

To run a spacific fairness testing approach include in this repository:
```bash
python fairbs.py --dataset_name census --sensitive_name age --classifier_name=dt
```

