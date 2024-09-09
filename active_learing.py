import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import margin_sampling
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def load_data():
    np.random.seed(0)
    n_samples = 2000
    X = pd.DataFrame({
        'usage_minutes': np.random.randint(1, 1000, n_samples),
        'contract_length': np.random.choice([1, 12, 24], n_samples),
        'monthly_charges': np.random.uniform(20, 100, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'customer_service_calls': np.random.poisson(2, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.uniform(20000, 100000, n_samples)
    })
    y = (np.random.rand(n_samples) < 0.3).astype(int)
    return X, y

def engineer_features(X):
    X['price_per_minute'] = X['monthly_charges'] / X['usage_minutes'].replace(0, 1)
    X['calls_to_usage_ratio'] = X['customer_service_calls'] / X['usage_minutes'].replace(0, 1)
    X['contract_value'] = X['monthly_charges'] * X['contract_length']
    X['income_per_minute'] = X['income'] / X['usage_minutes'].replace(0, 1)
    X['age_group'] = pd.cut(X['age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3])

    # Calculate mean usage before creating the categorical feature
    mean_usage = X['usage_minutes'].mean()
    X['high_usage'] = (X['usage_minutes'] > mean_usage).astype(int)

    # Convert 'contract_length' to categorical
    X['contract_length'] = X['contract_length'].astype('category')

    X = X.replace([np.inf, -np.inf], np.finfo(np.float64).max)

    for col in X.columns:
        if X[col].dtype != 'category':
            X[col] = X[col].clip(lower=np.finfo(np.float64).min, upper=np.finfo(np.float64).max)

    return X


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def initialize_committee(X_train, y_train):
    n_initial = 200
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

    X_initial = X_train.iloc[initial_idx]
    y_initial = y_train.iloc[initial_idx]

    X_pool = X_train.drop(X_train.index[initial_idx])
    y_pool = y_train.drop(y_train.index[initial_idx])

    learner1 = ActiveLearner(
        estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
        X_training=X_initial, y_training=y_initial,
        query_strategy=margin_sampling
    )

    learner2 = ActiveLearner(
        estimator=RandomForestClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=2),
        X_training=X_initial, y_training=y_initial,
        query_strategy=margin_sampling
    )

    committee = Committee(
        learner_list=[learner1, learner2],
        query_strategy=margin_sampling
    )

    return committee, X_pool, y_pool


def active_learning_loop(committee, X_pool, y_pool, X_test, y_test, n_queries=100, batch_size=10):
    performance_history = []

    for idx in range(n_queries):
        query_idx, query_instance = committee.query(X_pool, n_instances=batch_size)

        X_queried = X_pool.iloc[query_idx]
        y_queried = y_pool.iloc[query_idx]

        committee.teach(X_queried, y_queried)

        X_pool = X_pool.drop(X_pool.index[query_idx])
        y_pool = y_pool.drop(y_pool.index[query_idx])

        y_pred = committee.predict(X_test)
        performance_history.append(accuracy_score(y_test, y_pred))

        if (idx + 1) % 10 == 0:
            print(f'Query {idx + 1} completed. Current accuracy: {performance_history[-1]:.4f}')

    return performance_history


if __name__ == "__main__":
    X, y = load_data()
    X = engineer_features(X)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    committee, X_pool, y_pool = initialize_committee(X_train, y_train)
    performance_history = active_learning_loop(committee, X_pool, y_pool, X_test, y_test, n_queries=100, batch_size=10)

    print(f'Initial model accuracy: {performance_history[0]:.4f}')
    print(f'Final model accuracy: {performance_history[-1]:.4f}')
    print(f'Improvement: {performance_history[-1] - performance_history[0]:.4f}')

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(performance_history) + 1), performance_history)
    plt.xlabel('Number of queries')
    plt.ylabel('Accuracy')
    plt.title('Active Learning Performance (Ensemble Method)')
    plt.show()