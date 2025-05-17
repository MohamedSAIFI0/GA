import numpy as np
import matplotlib.pyplot as plt

# Génération de données linéaires simulées
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 1) * 10
true_w = 2.5
Y = true_w * X + np.random.randn(n_samples, 1) * 2  # bruit ajouté

# Paramètres GA
population_size = 30
n_generations = 100
p_crossover = 0.8
p_mutation = 0.1
mutation_strength = 3

# Initialisation aléatoire des poids (1 seul poids ici car X est en 1D)
population = np.random.uniform(-10, 10, (population_size, 1))

# Fonction de coût : erreur absolue moyenne
def cost(w):
    predictions = X @ w
    return np.mean(np.abs(predictions - Y))

# Évolution
best_scores = []

for generation in range(n_generations):
    # Évaluation
    fitness = np.array([cost(ind.reshape(-1, 1)) for ind in population])
    best_scores.append(fitness.min())
    
    # Sélection (roulette inversée sur le coût → meilleur = plus petit coût)
    inv_fitness = 1 / (fitness + 1e-6)  # éviter la division par zéro
    probs = inv_fitness / inv_fitness.sum()
    selected = population[np.random.choice(population_size, size=population_size, p=probs)]
    
    # Croisement (simple moyenne pondérée ici)
    offspring = []
    for i in range(0, population_size, 2):
        p1, p2 = selected[i], selected[(i+1) % population_size]
        if np.random.rand() < p_crossover:
            alpha = np.random.rand()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = alpha * p2 + (1 - alpha) * p1
        else:
            child1, child2 = p1.copy(), p2.copy()
        offspring += [child1, child2]

    # Mutation
    for i in range(population_size):
        if np.random.rand() < p_mutation:
            offspring[i] += np.random.randn(1) * mutation_strength

    # Mise à jour de la population
    population = np.clip(np.array(offspring), -100, 100)

# Meilleur individu
best_individual = population[np.argmin([cost(ind.reshape(-1, 1)) for ind in population])]
print(f"Meilleur poids trouvé : w = {best_individual[0]:.4f}")

# 🔍 Affichage des résultats
plt.plot(best_scores)
plt.xlabel("Itérations")
plt.ylabel("Coût (erreur absolue moyenne)")
plt.title("Évolution de l'erreur avec l'algorithme génétique")
plt.grid(True)
plt.show()
