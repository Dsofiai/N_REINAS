# -*- coding: utf-8 -*-
"""
Adaptado por Danna Sofía Imbachi
Basado en el código del profesor Camilo
"""

import numpy as np
import matplotlib.pyplot as plt

# Función de evaluación para N-Reinas: contar conflictos entre reinas
def fitness_function(x):
    conflicts = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            # Conflicto en diagonal
            if abs(x[i] - x[j]) == abs(i - j):
                conflicts += 1
    return conflicts  # Se busca minimizar conflictos

# Estrategia Evolutiva (μ, λ)
def evolutionary_strategy(mu, lambd, dim, generations, sigma=0.1):
    # Cada individuo es una permutación (no hay conflictos en filas ni columnas)
    def generate_individual():
        return np.random.permutation(dim)

    population = np.array([generate_individual() for _ in range(mu)])
    best_fitness_history = []

    print("\n Población Inicial:")
    print(population)

    for gen in range(generations):
        print(f"\n Generación {gen + 1}")

        selected_parents = []
        offspring = []

        for _ in range(lambd):
            parent_index = np.random.randint(mu)
            parent = population[parent_index].copy()

            # Mutación por intercambio de dos posiciones
            i, j = np.random.choice(dim, 2, replace=False)
            parent[i], parent[j] = parent[j], parent[i]

            offspring.append(parent)
            selected_parents.append(parent)

        offspring = np.array(offspring)

        print(f"Padres seleccionados ({len(selected_parents)} en total):")
        for i, parent in enumerate(selected_parents[:5]):  # mostrar solo los primeros 5
            print(f"   Padre {i + 1}: {parent}")

        # Evaluar aptitud
        fitness_values = np.array([fitness_function(ind) for ind in offspring])
        best_indices = np.argsort(fitness_values)[:mu]
        population = offspring[best_indices]

        best_fitness = fitness_values[best_indices[0]]
        best_fitness_history.append(best_fitness)
        print(f"Mejor aptitud en esta generación = {best_fitness}")
        print(f" Mejor Individuo: {population[0]}")

        # Detener si se encuentra solución perfecta
        if best_fitness == 0:
            print("\n¡Solución válida encontrada!")
            break

    # Graficar evolución
    plt.figure(figsize=(8, 5))
    plt.plot(best_fitness_history, marker='o', linestyle='-', color='b', label="Mejor Aptitud")
    plt.xlabel("Generaciones")
    plt.ylabel("Conflictos entre reinas")
    plt.title("Convergencia en el problema de N-Reinas")
    plt.legend()
    plt.grid()
    plt.show()

    best_solution = population[0]
    return best_solution, fitness_function(best_solution)

# Parámetros
mu = 10           # Tamaño de la población
lambd = 40        # Número de descendientes
dim = 8           # Número de reinas (N)
generations = 100 # Generaciones
sigma = 0.1       # No se usa directamente en esta versión

# Ejecutar algoritmo
best_sol, best_fitness = evolutionary_strategy(mu, lambd, dim, generations, sigma)

print("\n Mejor solución encontrada:", best_sol)
print(" Conflictos en la solución:", best_fitness)

