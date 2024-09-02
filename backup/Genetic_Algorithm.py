import time

import numpy as np
from sklearn.inspection import permutation_importance


class GA:
    def __str__(self):
        return "GA"

    def __init__(self, model, population, DNA_SIZE, bound, fitness_func, X, Y, cross_rate=0.8, mutation_rate=0.003):
        self.model = model
        self.population = population
        self.pop_size = len(population)
        self.DNA_SIZE = DNA_SIZE
        self.bound = np.array(bound)
        self.fitness_func = fitness_func
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.feature_importance = None
        self.importance_threshold = 0.05
        self.diversity_weight = 0.3

        self.tournament_size = max(5, int(self.pop_size * 0.05))

        self.analyze_feature_importance(X, Y)

    def analyze_feature_importance(self, X, y):
        result = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)
        self.feature_importance = result.importances_mean / np.sum(result.importances_mean)

    def calculate_diversity(self):
        return np.mean(np.std(self.population, axis=0))

    def get_fitness(self):
        base_fitness = np.array([self.fitness_func(self.population[i]) for i in range(len(self.population))])
        diversity = self.calculate_diversity()
        combined_fitness = base_fitness + self.diversity_weight * diversity
        return combined_fitness

    def select(self):
        fitness = self.get_fitness()
        selection_probs = fitness / np.sum(fitness)

        selected = []
        for _ in range(self.pop_size):
            tournament = np.random.choice(
                np.arange(self.pop_size),
                self.tournament_size,
                replace=False,
                p=selection_probs
            )
            winner = tournament[np.argmax(fitness[tournament])]
            selected.append(self.population[winner])

        self.population = np.vstack(selected)

    def crossover(self):
        for i in range(0, self.pop_size, 2):
            if np.random.rand() < self.cross_rate:
                partner_idx = np.random.randint(0, self.pop_size)
                for j in range(self.DNA_SIZE):
                    if self.feature_importance[j] > self.importance_threshold:
                        if np.random.rand() < 0.5:
                            self.population[i, j], self.population[partner_idx, j] = \
                                self.population[partner_idx, j], self.population[i, j]

    def mutate(self):
        for i in range(self.pop_size):
            for j in range(self.DNA_SIZE):
                if np.random.rand() < self.mutation_rate:
                    if self.feature_importance[j] > self.importance_threshold:
                        mutation_range = (self.bound[j, 1] - self.bound[j, 0]) * self.feature_importance[j]
                        self.population[i, j] += np.random.uniform(-mutation_range, mutation_range)
                    else:
                        self.population[i, j] = np.random.uniform(self.bound[j, 0], self.bound[j, 1])

                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])

    def evolve(self):
        self.select()
        self.crossover()
        self.mutate()
