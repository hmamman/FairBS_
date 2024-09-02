import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random


class GeneticAlgorithmCounterfactual:
    def __init__(self, data, model, categorical_features=None, population_size=100, generations=50):
        self.data = data
        self.model = model
        self.categorical_features = categorical_features or []
        self.numerical_features = [col for col in data.columns if col not in self.categorical_features]
        self.population_size = population_size
        self.generations = generations

        # Create preprocessor
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Fit the preprocessor
        self.preprocessor.fit(self.data)

        # Store the feature names and their respective ranges
        self.feature_names = (self.numerical_features +
                              self.preprocessor.named_transformers_['cat'].get_feature_names_out(
                                  self.categorical_features).tolist())
        self.feature_ranges = self._get_feature_ranges()

    def _get_feature_ranges(self):
        ranges = {}
        for i, feature in enumerate(self.numerical_features):
            ranges[i] = (self.data[feature].min(), self.data[feature].max())

        cat_start = len(self.numerical_features)
        for i, feature in enumerate(self.categorical_features):
            cat_range = self.preprocessor.named_transformers_['cat'].categories_[i]
            one_hot_start = cat_start + sum(
                len(c) for c in self.preprocessor.named_transformers_['cat'].categories_[:i])
            for j, category in enumerate(cat_range):
                ranges[one_hot_start + j] = (0, 1)
        return ranges

    def transform_instance(self, instance):
        if not isinstance(instance, pd.DataFrame):
            instance = pd.DataFrame([instance], columns=self.data.columns)
        return self.preprocessor.transform(instance)

    def inverse_transform_instance(self, instance):
        num_features = len(self.numerical_features)
        cat_features = len(self.categorical_features)

        # Inverse transform numerical features
        num_inverse = self.preprocessor.named_transformers_['num'].inverse_transform(instance[:, :num_features])

        # Inverse transform categorical features
        cat_inverse = []
        start = num_features
        for i, feature in enumerate(self.categorical_features):
            end = start + len(self.preprocessor.named_transformers_['cat'].categories_[i])
            cat_data = instance[:, start:end]
            cat_inverse.append(self.preprocessor.named_transformers_['cat'].categories_[i][cat_data.argmax(axis=1)])
            start = end

        # Combine numerical and categorical features
        inverse_data = np.column_stack([num_inverse] + cat_inverse)
        return pd.DataFrame(inverse_data, columns=self.numerical_features + self.categorical_features)

    def predict(self, instance):
        if isinstance(instance, np.ndarray):
            instance = self.inverse_transform_instance(instance)
        return self.model.predict(instance)[0]

    def fitness(self, individual, original, desired_outcome):
        try:
            prediction = self.predict(individual.reshape(1, -1))
            distance = np.sum((original - individual) ** 2)
            outcome_penalty = 0 if prediction == desired_outcome else 1e6
            return distance + outcome_penalty
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return float('inf')

    def crossover(self, parent1, parent2):
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def mutate(self, individual, mutation_rate=0.1):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                if i in self.feature_ranges:
                    if self.feature_ranges[i][1] - self.feature_ranges[i][0] == 1:  # binary feature
                        individual[i] = 1 - individual[i]
                    else:
                        individual[i] += np.random.normal(0,
                                                          0.1 * (self.feature_ranges[i][1] - self.feature_ranges[i][0]))
                        individual[i] = np.clip(individual[i], self.feature_ranges[i][0], self.feature_ranges[i][1])
        return individual

    def generate_counterfactuals(self, instance, desired_outcome, n_counterfactuals=1):
        original = self.transform_instance(instance).flatten()

        population = [self.mutate(original.copy(), mutation_rate=0.5) for _ in range(self.population_size)]

        for _ in range(self.generations):
            fitness_scores = [self.fitness(ind, original, desired_outcome) for ind in population]

            parents = [population[i] for i in np.argsort(fitness_scores)[:self.population_size // 2]]

            next_generation = parents.copy()
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation

        fitness_scores = [self.fitness(ind, original, desired_outcome) for ind in population]
        best_indices = np.argsort(fitness_scores)[:n_counterfactuals]
        counterfactuals = [population[i] for i in best_indices]

        return [self.inverse_transform_instance(cf.reshape(1, -1)) for cf in counterfactuals]


# Test function
def test_genetic_algorithm_counterfactual():
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.randint(20000, 200000, 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'employed': np.random.choice([0, 1], 1000)
    })
    data['loan_approved'] = ((data['credit_score'] > 700) & (data['income'] > 50000) & (data['age'] > 25)).astype(int)

    # Split data and train a model
    X = data.drop('loan_approved', axis=1)
    y = data['loan_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a preprocessing pipeline for the model
    categorical_features = ['gender', 'employed']
    numerical_features = ['age', 'income', 'credit_score']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    # Initialize GeneticAlgorithmCounterfactual
    ga_cf = GeneticAlgorithmCounterfactual(X_train, model, categorical_features=categorical_features)

    # Generate counterfactuals for a specific instance
    instance = X_test.iloc[0]
    desired_outcome = 1 - model.predict(instance.to_frame().T)[0]
    n_counterfactuals = 5

    try:
        counterfactuals = ga_cf.generate_counterfactuals(instance, desired_outcome, n_counterfactuals=n_counterfactuals)

        print("Original instance:")
        print(instance)
        print(f"Original prediction: {model.predict(instance.to_frame().T)[0]}")

        print(f"\nGenerated {len(counterfactuals)} counterfactuals:")
        for i, cf in enumerate(counterfactuals, 1):
            print(f"\nCounterfactual {i}:")
            print(cf.iloc[0])
            print(f"Prediction: {model.predict(cf)[0]}")

        print("\nTest passed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")


# Run the test
if __name__ == "__main__":
    test_genetic_algorithm_counterfactual()