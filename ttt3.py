import numpy as np
import matplotlib.pyplot as plt

def fitness_function(solution):
    # Cuenta el número de pares de reinas que se atacan
    n = len(solution)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(solution[i] - solution[j]) == abs(i - j):
                conflicts += 1
    return conflicts

def crossover(parent1, parent2):
    n = len(parent1)
    point = np.random.randint(1, n - 1)
    child = np.empty(n, dtype=int)
    child[:point] = parent1[:point]
    pointer = point
    for gene in parent2:
        if gene not in child[:point]:
            child[pointer] = gene
            pointer += 1
    return child

def mutate(solution, mutation_rate):
    n = len(solution)
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(n, 2, replace=False)
        solution[i], solution[j] = solution[j], solution[i]
    return solution

def evolutionary_strategy(mu, lambd, N, generations, sigma):
    # Inicializar población
    population = [np.random.permutation(N) for _ in range(mu)]
    fitness_values = [fitness_function(ind) for ind in population]
    best_fitness_history = []

    for generation in range(generations):
        print(f"\nGeneración {generation + 1}")
        offspring = []

        for _ in range(lambd):
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)
            p1, p2 = population[idx1], population[idx2]
            print(f"Padre 1: {p1}")
            print(f"Padre 2: {p2}")

            child = crossover(p1, p2)
            child = mutate(child.copy(), sigma)
            offspring.append(child)

        # Evaluar aptitud de descendientes
        offspring_fitness = [fitness_function(ind) for ind in offspring]

        # Selección de los mejores individuos
        combined = population + offspring
        combined_fitness = fitness_values + offspring_fitness
        best_indices = np.argsort(combined_fitness)[:mu]
        population = [combined[i] for i in best_indices]
        fitness_values = [combined_fitness[i] for i in best_indices]

        best_fitness = fitness_values[0]
        best_fitness_history.append(best_fitness)
        print(f"Mejor aptitud en esta generación = {best_fitness}")
        print(f"Mejor Individuo: {population[0]}")

        if best_fitness == 0:
            print("\n¡Solución válida encontrada!")
            break

    # Graficar evolución
    plt.figure(figsize=(8, 5))
    plt.plot(best_fitness_history, marker='o', linestyle='-', label="Mejor Aptitud")
    plt.xlabel("Generaciones")
    plt.ylabel("Conflictos entre reinas")
    plt.title("Convergencia en el problema de N-Reinas")
    plt.legend()
    plt.grid()
    plt.show()

    best_solution = population[0]
    return best_solution, fitness_function(best_solution)

# Parámetros
mu = 10             # Tamaño de la población
lambd = 40          # Número de descendientes
dim = 8             # Número de reinas (N)
generations = 100   # Número de generaciones
sigma = 0.1         # Tasa de mutación

# Ejecutar algoritmo
best_sol, best_fitness = evolutionary_strategy(mu, lambd, dim, generations, sigma)

print("\nMejor solución encontrada:", best_sol)
print("Conflictos en la solución:", best_fitness)

