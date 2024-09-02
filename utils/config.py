class census:
    """
    Configuration of dataset Census Income
    """
    dataset_name = 'census'
    # the size of total features
    params = 13
    sensitive_param = [9, 1, 8]

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39])  # 69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race",
                    "sex", "capital_gain",
                    "capital_loss", "hours_per_week", "native_country"]

    sens_name = {9: 'sex', 1: "age", 8: "race"}  # Starts at 1

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]


class credit:
    """
    Configuration of dataset German Credit
    """
    dataset_name = 'credit'
    # the size of total features
    params = 20
    sensitive_param = 9

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status",
                    "employment", "installment_commitment", "sex", "other_parties",
                    "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits",
                    "job", "num_dependents", "own_telephone", "foreign_worker"]

    sens_name = {9: 'sex', 13: "age"}  # Starts at 1

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    initial_input = [2, 24, 2, 2, 37, 0, 1, 2, 1, 0, 4, 2, 2, 2, 1, 1, 2, 1, 0, 0]


class bank:
    """
    Configuration of dataset Bank Marketing
    """
    dataset_name = 'bank'
    # the size of total features
    params = 16
    sensitive_param = 1

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                    "month", "duration", "campaign", "pdays", "previous", "poutcome"]
    sens_name = {1: "age", 3: "marital"}  # Starts at 1

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    initial_input = [1, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 45, 30, 2]


class compas:
    dataset_name = 'compas'
    # the size of total features
    params = 14

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([1, 9])
    input_bounds.append([0, 5])
    input_bounds.append([0, 20])
    input_bounds.append([0, 13])
    input_bounds.append([0, 11])
    input_bounds.append([0, 4])
    input_bounds.append([-6, 10])
    input_bounds.append([0, 90])
    input_bounds.append([0, 2])
    input_bounds.append([-1, 1])
    input_bounds.append([0, 2])
    input_bounds.append([0, 1])
    input_bounds.append([-1, 10])

    # the name of each feature
    feature_name = ["sex",
                    "age",
                    "race",
                    "juv_fel_count",
                    "juv_misd_count",
                    "juv_other_count",
                    "priors_count",
                    "days_b_screening_arrest",
                    "c_days_from_compas",
                    "c_charge_degree",
                    "is_recid",
                    "r_charge_degree",
                    "is_violent_recid",
                    "v_decile_score"]

    sens_name = {1: 'sex', 2: 'age', 3: 'race'}  # Starts at 1
    # the name of each class
    class_name = ["Low", "High"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    initial_input = [1, 1, 1, 0, 0, 0, 2, -2, 2, 0, 0, 1, 0, 10]


class meps:
    """
    Configuration of dataset meps
    """
    dataset_name = 'meps'
    # the size of total features
    params = 40

    # the valid religion of each feature
    input_bounds = [
        [0, 3],
        [0, 85],
        [0, 1],
        [0, 1],
        [0, 9],
        [0, 3],
        [0, 3],
        [0, 3],
        [0, 5],
        [0, 5],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 3],
        [0, 1],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [-9, 70],
        [-9, 75],
        [-9, 24],
        [0, 7],
        [0, 4],
        [0, 4],
        [0, 2],
    ]
    # the name of each feature
    feature_name = [
        'REGION',
        'AGE',
        'SEX',
        'RACE',
        'MARRY',
        'FTSTU',
        'ACTDTY',
        'HONRDC',
        'RTHLTH',
        'MNHLTH',
        'CHDDX',
        'ANGIDX',
        'MIDX',
        'OHRTDX',
        'STRKDX',
        'EMPHDX',
        'CHBRON',
        'CHOLDX',
        'CANCERDX',
        'DIABDX',
        'JTPAIN',
        'ARTHDX',
        'ARTHTYPE',
        'ASTHDX',
        'ADHDADDX',
        'PREGNT',
        'WLKLIM',
        'ACTLIM',
        'SOCLIM',
        'COGLIM',
        'DFHEAR42',
        'DFSEE42',
        'ADSMOK42',
        'PCS42',
        'MCS42',
        'K6SUM42',
        'PHQ242',
        'EMPST',
        'POVCAT',
        'INSCOV',
    ]

    sens_name = {3: 'SEX'}
    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = range(0, 40)

    initial_input = [3, 59, 0, 1, 1, 1, 0, 0, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
                     -1, -1, -1, 1, 3, 3, 1]
