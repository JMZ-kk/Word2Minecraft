import numpy as np
import heapq
import random
import math
import matplotlib.pyplot as plt
import json

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return g_score[current]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] != 1:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return float('inf')


def find_distances(grid):
    objectives = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == 2]
    if len(objectives) != 8:
        return float('inf')
    total_distance = 0
    for i in range(len(objectives)):
        for j in range(i + 1, len(objectives)):
            dist = a_star(grid, objectives[i], objectives[j])
            if math.isinf(dist):
                return float('inf')
            total_distance += dist
    return total_distance / (len(objectives) * (len(objectives) - 1) / 2)


def is_valid_map(grid):
    return np.count_nonzero(grid == 3) == 1 and np.count_nonzero(grid == 2) == 8


def fitness(grid):
    if not is_valid_map(grid):
        return -1
    avg_distance = find_distances(grid)
    return avg_distance if not math.isinf(avg_distance) else -1


def generate_random_map(size):
    grid = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
    objectives = random.sample([(r, c) for r in range(size) for c in range(size) if grid[r, c] == 0], 8)
    start = random.choice(
        [(r, c) for r in range(size) for c in range(size) if grid[r, c] == 0 and (r, c) not in objectives])
    for obj in objectives:
        grid[obj] = 2
    grid[start] = 3
    return grid


def crossover(parent1, parent2):
    size = parent1.shape[0]
    point = random.randint(1, size - 2)
    child1 = np.vstack((parent1[:point], parent2[point:]))
    child2 = np.vstack((parent2[:point], parent1[point:]))
    return child1, child2


def mutate(grid, mutation_rate=0.1):
    size = grid.shape[0]
    for _ in range(int(size * size * mutation_rate)):
        r, c = random.randint(0, size - 1), random.randint(0, size - 1)
        if grid[r, c] not in [2, 3]:
            grid[r, c] = random.choice([0, 1])
    return grid


def genetic_algorithm(pop_size=50, generations=100, size=15):
    population = [generate_random_map(size) for _ in range(pop_size)]
    fitness_history = []
    for gen in range(generations):
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        fitness_history.append(fitness(population[0]))
        new_population = population[:pop_size // 4]
        while len(new_population) < pop_size:
            p1, p2 = random.choices(population[:pop_size // 2], k=2)
            c1, c2 = crossover(p1, p2)
            new_population.extend([mutate(c1), mutate(c2)])
        population = new_population[:pop_size]
    return population[0], fitness_history


def run_experiments(num_experiments=30, generations=150):
    all_histories = []
    for tt in range(num_experiments):
        best, history = genetic_algorithm(generations=generations)
        with open(f"data_{tt}.json", "w") as f:
            json.dump(best.tolist(), f)
        all_histories.append(history)
    right_column = [row[-1] for row in all_histories]
    avg_fitness_per_gen = np.mean(all_histories, axis=0)
    print(np.mean(right_column))
    print(np.std(right_column))
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(10, 5))
    plt.plot(range(generations), avg_fitness_per_gen, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('ASPAO')
    plt.title('ASPAO Evaluation')
    plt.legend()
    plt.show()


run_experiments()
