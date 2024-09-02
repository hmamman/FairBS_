import copy
import time

import numpy as np
import pandas as pd
import random


class GA:
    def __init__(self, pop_size, num_vars, bounds, fitness_func, group_fitness, dna_size=None, cross_rate=0.8,
                 mutation_rate=0.003, local_generation=None, initial_input=None, sensitive_param=None):
        self.pop_size = pop_size
        self.population = None
        self.best_solutions = None
        self.best_fitness = -1
        self.num_vars = num_vars
        self.bounds = np.array(bounds)
        self.fitness_func = fitness_func
        self.initial_input = initial_input
        self.local_generation = local_generation
        self.group_fitness = group_fitness
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.fitness_threshold = np.finfo(float).eps
        self.sensitive_param = sensitive_param

        min_vals, max_vals = np.array(list(zip(*self.bounds)))
        var_ranges = max_vals - min_vals

        if dna_size is None:
            dna_size = int(np.ceil(np.log2(np.max(var_ranges) + 1)))
        self.dna_size = dna_size

        self.generate_population()
        self.best_population = self.population.copy()

    def _encode(self, population, min_vals, var_ranges, dna_size):
        encoded_pop = np.zeros((*population.shape, dna_size), dtype=int)  # Set dtype to int
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                num = int(round((population[i, j] - min_vals[j]) * ((2 ** dna_size) / var_ranges[j])))
                encoded_pop[i, j] = [int(k) for k in ('{0:0' + str(dna_size) + 'b}').format(num)]
        return encoded_pop

    def _decode(self, encoded_pop, min_vals, var_ranges, dna_size):
        weight_vector = np.array([2 ** i for i in range(dna_size)])[::-1]
        binary_vector = np.zeros((encoded_pop.shape[0], encoded_pop.shape[1]))

        if encoded_pop.dtype != int:
            encoded_pop = encoded_pop.astype(int)  # Convert to int if not already
        for i in range(encoded_pop.shape[0]):
            for j in range(encoded_pop.shape[1]):
                binary_value = 0
                for k in range(self.dna_size):
                    binary_value += encoded_pop[i, j] & (1 << k)
                binary_vector[i, j] = binary_value / ((2 ** self.dna_size) / var_ranges[j]) + min_vals[j]

        return binary_vector

    def global_discovery(self, x):
        random.seed(time.time())
        for i in range(len(x)):
            x[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
            if i == self.sensitive_param - 1:
                x[i] = self.bounds[i][0]
            self.population = np.append(self.population, [x], axis=0)
        return x

    def generate_population(self):
        low_bounds = np.array([bound[0] for bound in self.bounds])
        high_bounds = np.array([bound[1] for bound in self.bounds])

        low_bounds = np.tile(low_bounds, (self.pop_size, 1))
        high_bounds = np.tile(high_bounds, (self.pop_size, 1))

        self.population = np.random.uniform(low=low_bounds, high=high_bounds, size=(self.pop_size, self.num_vars))
        return self.population

    def get_fitness(self, non_negative=False):
        decoded_pop = self._decode(self.population, *np.array(list(zip(*self.bounds))), self.dna_size)
        fitness = [self.fitness_func(decoded_pop[i]) for i in range(len(decoded_pop))]

        # # Handle potential division by zero and NaN values (adjust based on your fairness metric)
        if np.isnan(fitness).any():
            fitness[np.isnan(fitness)] = 0.0  # Assign low fitness for NaN values
        fitness = fitness + self.fitness_threshold  # Add epsilon to avoid division by zero
        # # fitness = fitness / np.sum(fitness)
        return fitness

    def expand_population(self, max_attempts=5000):
        initial_size = len(self.population)
        attempts = 0

        while len(self.population) < self.pop_size and attempts < max_attempts:
            new_individuals = []

            for inp in self.population:
                if len(self.population) + len(new_individuals) >= self.pop_size:
                    break

                for param in range(len(inp)):
                    for direction in [-1, 1]:
                        inp2 = copy.copy(inp)
                        inp2[param] = inp2[param] + direction

                        if param == self.sensitive_param - 1:  # Only perturb the non-sensitive parameters
                            continue

                        # Check bounds
                        if inp2[param] < self.bounds[param][0] or inp2[param] > self.bounds[param][1]:
                            continue

                        # Check if the new individual is unique
                        if not any(np.array_equal(inp2, existing_inp) for existing_inp in self.population) and \
                                not any(np.array_equal(inp2, new_ind) for new_ind in new_individuals):
                            new_individuals.append(inp2)

                            if len(self.population) + len(new_individuals) >= self.pop_size:
                                break

                    if len(self.population) + len(new_individuals) >= self.pop_size:
                        break

            # Add new individuals to the population
            self.population = np.vstack((self.population, new_individuals))

            # Increment attempt counter
            attempts += 1

        # If we couldn't reach pop_size, randomly duplicate existing individuals
        if len(self.population) < self.pop_size:
            additional_needed = self.pop_size - len(self.population)
            additional_individuals = self.population[
                np.random.choice(len(self.population), size=additional_needed, replace=True)]
            self.population = np.vstack((self.population, additional_individuals))

        print(f"Population expanded from {initial_size} to {len(self.population)} individuals in {attempts} attempts.")

    def select(self):
        fitness = self.get_fitness()
        # filtered_population = self.local_generation(initial_input=self.initial_input,
        #                                             global_discovery=self.global_discovery)
        # total_filtered_population = len(filtered_population)
        # self.population = filtered_population
        # self.expand_population()
        # if total_filtered_population > 0:
        #     if total_filtered_population < self.pop_size:
        #
        #         while len(self.population) < self.pop_size:
        #             for inp in self.population:
        #                 for param in range(len(inp)):
        #                     # if param == self.sensitive_param - 1:  # Only perturb the non-sensitive parameters
        #                     #     continue
        #                     for direction in [-1, 1]:
        #                         inp2 = copy.copy(inp)
        #                         inp2[param] = inp2[param] + direction
        #                         if inp2[param] < self.bounds[param][0] and direction == -1:
        #                             continue
        #                         elif inp2[param] > self.bounds[param][1] and direction == 1:
        #                             continue
        #
        #                         if inp2 not in self.population:
        #                             self.population = np.append(self.population, [inp2], axis=0)

        # else:
        #     self.population = filtered_population[
        #         np.random.choice(np.arange(len(filtered_population)), size=self.pop_size, replace=True)
        #         #     len(filtered_population)
        #     ]
        # # print(self.population)
        # # print(filtered_population)
        # self.population = filtered_population[
        #     np.random.choice(np.arange(len(filtered_population)), size=self.pop_size, replace=True)
        #     #     len(filtered_population)
        # ]
        # print(self.population)
        # print(filtered_population)

        # # Filter candidates with fitness > 0
        # # Get indices of positive fitness that are greater than epsilon
        # positive_fitness_indices = np.where(np.array(fitness) > self.fitness_threshold)[0]
        #
        # filtered_population = self.population[positive_fitness_indices]  # Select corresponding candidates
        #
        # Check if any candidates with positive fitness exist
        # if len(filtered_population) > 0:
        #     self.population = filtered_population[
        #         np.random.choice(np.arange(len(filtered_population)), size=self.pop_size, replace=True)
        #         #     len(filtered_population)
        #     ]
        if len(filtered_population) > 0:
            # self.population = [np.random.choice(np.arange(len(filtered_population)), size=self.pop_size, replace=True)]
            self.population = np.array(filtered_population)
        else:
            # No candidates with positive fitness, use original selection logic
            self.population = self.population[
                np.random.choice(np.arange(self.population.shape[0]), size=self.population.shape[0], replace=True,
                                 p=fitness / np.sum(fitness))]

    def crossover(self):
        for i in range(self.population.shape[0]):
            if np.random.rand() < self.cross_rate:
                partner_idx = np.random.randint(0, self.population.shape[0])
                cross_points = np.random.randint(0, self.num_vars)
                self.population[i, :cross_points] = self.population[partner_idx, :cross_points]

    def mutate(self):
        for i in range(self.population.shape[0]):
            for j in range(self.population.shape[1]):
                if np.random.rand() < self.mutation_rate:
                    self.population[i, j] = np.random.uniform(low=self.bounds[j][0],
                                                              high=self.bounds[j][1])  # Mutate variable directly

    def evolve(self, num_generations):
        for _ in range(num_generations):
            self.select()
            self.crossover()
            self.mutate()
            group_fitness = self.group_fitness(self.population)
            if group_fitness > self.best_fitness:
                self.best_fitness = group_fitness
                self.best_solutions = self.population.copy()

    def get_best_solution(self):
        return self._decode(self.best_population, *np.array(list(zip(*self.bounds))), self.dna_size)[0]

    # def regenerate_solutions(self, solutions, size):
    #     solutions = np.array(solutions)
    #
    #     count = len(solutions)
    #     for inp in solutions:
    #         if count >= size:
    #             break
    #
    #         for param in range(len(inp)):
    #             # if param == self.sensitive_param - 1:  # Only perturb the non-sensitive parameters
    #             #     continue
    #             for direction in [-1, 1]:
    #                 inp2 = inp.copy()
    #                 inp2[param] = inp2[param] + direction
    #                 if inp2[param] < self.bounds[param][0] and direction == -1:
    #                     continue
    #                 elif inp2[param] > self.bounds[param][1] and direction == 1:
    #                     continue
    #
    #                 if inp2 not in solutions:
    #                     solutions = np.append(solutions, inp2)
    #                     count += 1
    #     return solutions

    # def regenerate_solutions(self, solutions, size):
    #     solutions = np.array(solutions)
    #     solution_set = set(solutions)
    #     count = len(solutions)
    #
    #     while count < size:
    #         # Randomly select a solution from the existing solutions
    #         inp = random.choice(solutions)
    #
    #         # Randomly select a subset of parameters to perturb
    #         num_params_to_perturb = random.randint(1, len(inp) // 2)  # Adjust this range as needed
    #         params_to_perturb = random.sample(range(len(inp)), num_params_to_perturb)
    #
    #         for param in params_to_perturb:
    #             # Randomly perturb the selected parameter within the bounds
    #             direction = random.choice([-1, 1])
    #             inp[param] = max(self.bounds[param][0], min(self.bounds[param][1], inp[param] + direction))
    #
    #         # Add the perturbed solution if it's new and within the bounds
    #         inp2 = tuple(inp)
    #         if tuple(map(tuple, inp2)) not in solution_set:
    #             solution_set.add(tuple(map(tuple, inp2)))
    #             solutions = np.append(solutions, inp, axis=0)
    #             count += 1
    #         print(count)
    #
    #     return solutions[:size]
