import numpy as np

# Génération de données simulées
np.random.seed(0)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
true_w = np.array([2.0, -1.0, 0.5, 3.0, -2.0])
y = X @ true_w + np.random.normal(0, 1, n_samples)

# Fonction de coût : Erreur absolue moyenne (L1)
def l1_loss(w, X, y):
    y_pred = X @ w
    return np.mean(np.abs(y_pred - y))

# Initialiser une population réelle
def initialize_population(size, n_genes):
    return np.random.uniform(-5, 5, (size, n_genes))

# Croisement (crossover)
def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

# Mutation
def mutate(ind, mutation_rate=0.1):
    noise = np.random.randn(*ind.shape) * mutation_rate
    return ind + noise

# Sélection par tournoi
def tournament_selection(pop, fitness, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmin(fitness[idx])]]

# Algorithme génétique
def genetic_algorithm(X, y, pop_size=30, n_gen=100, mutation_rate=0.1, crossover_rate=0.8):
    n_genes = X.shape[1]
    population = initialize_population(pop_size, n_genes)

    for gen in range(n_gen):
        fitness = np.array([l1_loss(ind, X, y) for ind in population])
        new_population = []

        for _ in range(pop_size):
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)

            if np.random.rand() < crossover_rate:
                child = crossover(p1, p2)
            else:
                child = p1.copy()

            if np.random.rand() < mutation_rate:
                child = mutate(child, mutation_rate)

            new_population.append(child)

        population = np.array(new_population)

        if gen % 10 == 0:
            print(f"Génération {gen} | Erreur L1 = {fitness.min():.4f}")

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

# Exécution
best_w, best_loss = genetic_algorithm(X, y)
print("\n Meilleur w trouvé :", best_w)
print(" Erreur L1 finale  :", best_loss)
print(" w réel             :", true_w)
