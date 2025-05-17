import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Charger les données depuis un fichier CSV
# Nous utilisons try/except pour gérer les différentes possibilités de formats
try:
    # Essayer de charger le fichier CSV
    df = pd.read_csv(r"C:\Users\P15\Desktop\Exerice\Advertising Budget and Sales.csv")
    
    # Vérifier si le fichier a un indice numérique comme première colonne qui n'est pas nécessaire
    if df.columns[0].isdigit() or df.columns[0] == "Unnamed: 0":
        # Si la première colonne est un indice, l'utiliser comme index et la supprimer des colonnes
        df = pd.read_csv(r"C:\Users\P15\Desktop\Exerice\Advertising Budget and Sales.csv", index_col=0)
except Exception as e:
    print(f"Erreur lors du chargement du fichier CSV: {e}")
    print("Veuillez vous assurer que le fichier 'mydata.csv' existe et a le format correct.")
    exit(1)

print("Aperçu des données chargées:")
print(df.head())

# Paramètres de l'algorithme génétique
Pm = 0.1      # Probabilité de mutation
Pc = 0.3      # Probabilité de croisement
Tmax = 100    # Nombre maximal d'itérations
N = 20        # Taille de la population
d = 1.2       # Paramètre de sélection
b = 3         # Paramètre de mutation

class GeneticAlgorithm:
    def __init__(self, X, Y, pop_size=N, pm=Pm, pc=Pc, max_iter=Tmax, d_param=d, b_param=b):
        self.X = X
        self.Y = Y
        self.n_samples = len(X)
        self.dim = X.shape[1]  # Dimension de l'espace des caractéristiques
        self.pop_size = pop_size
        self.pm = pm
        self.pc = pc
        self.max_iter = max_iter
        self.d_param = d_param
        self.b_param = b_param
        self.population = self.initialize_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def initialize_population(self):
        # Initialisation aléatoire de la population
        return [np.random.uniform(-1, 1, self.dim) for _ in range(self.pop_size)]
    
    def fitness(self, omega):
        # Fonction d'erreur E(ω) = (1/n)·Σ|ω·x⁽ⁱ⁾ - y⁽ⁱ⁾|
        predictions = np.dot(self.X, omega)
        error = np.mean(np.abs(predictions - self.Y))
        return error
    
    def evaluate_population(self):
        # Évaluation de la population
        fitness_values = [self.fitness(ind) for ind in self.population]
        return fitness_values
    
    def selection(self, fitness_values):
        # Sélection par roulette avec ajustement linéaire
        fitness_max = max(fitness_values)
        adjusted_fitness = [fitness_max - f for f in fitness_values]
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:  # Éviter la division par zéro
            probabilities = [1/len(adjusted_fitness) for _ in adjusted_fitness]
        else:
            probabilities = [f/total_fitness for f in adjusted_fitness]
        
        # Sélection avec d_param
        selected_indices = random.choices(range(self.pop_size), probabilities, k=self.pop_size)
        selected = [self.population[i] for i in selected_indices]
        return selected
    
    def crossover(self, parent1, parent2):
        # Croisement arithmétique
        if random.random() < self.pc:
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        return parent1, parent2
    
    def mutation(self, individual):
        # Mutation avec paramètre b
        if random.random() < self.pm:
            mutation_point = random.randint(0, self.dim - 1)
            mutation_value = random.uniform(-self.b_param, self.b_param)
            individual[mutation_point] += mutation_value
        return individual
    
    def evolve(self):
        # Évaluation des individus
        fitness_values = self.evaluate_population()
        
        # Mettre à jour la meilleure solution
        min_fitness_idx = np.argmin(fitness_values)
        if fitness_values[min_fitness_idx] < self.best_fitness:
            self.best_fitness = fitness_values[min_fitness_idx]
            self.best_solution = self.population[min_fitness_idx].copy()
        
        # Enregistrer l'historique des performances
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(np.mean(fitness_values))
        
        # Sélection
        selected = self.selection(fitness_values)
        
        # Création d'une nouvelle population
        new_population = []
        
        # Élitisme: conserver le meilleur individu
        new_population.append(self.population[min_fitness_idx].copy())
        
        # Appliquer le croisement et la mutation
        for i in range(0, self.pop_size - 1, 2):
            if i + 1 < self.pop_size:
                offspring1, offspring2 = self.crossover(selected[i], selected[i+1])
                new_population.append(self.mutation(offspring1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutation(offspring2))
        
        self.population = new_population
    
    def run(self):
        # Exécuter l'algorithme génétique
        for i in range(self.max_iter):
            self.evolve()
            
            # Afficher la progression
            if (i+1) % 10 == 0:
                print(f"Itération {i+1}/{self.max_iter}, Meilleure fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness_history, self.avg_fitness_history

def plot_results(ga, X_test=None, y_test=None):
    plt.figure(figsize=(15, 12))
    
    # Tracer l'évolution de la fitness
    plt.subplot(2, 2, 1)
    plt.plot(ga.best_fitness_history, 'b-', label="Meilleure fitness")
    plt.plot(ga.avg_fitness_history, 'r--', label="Fitness moyenne")
    plt.xlabel("Itération")
    plt.ylabel("Fitness (Erreur)")
    plt.title("Évolution de la fitness")
    plt.legend()
    plt.grid(True)
    
    # Coefficients
    plt.subplot(2, 2, 2)
    coeffs = ga.best_solution
    plt.bar(range(len(coeffs)), coeffs)
    plt.xlabel("Caractéristique")
    plt.ylabel("Coefficient")
    plt.title("Coefficients du modèle")
    plt.xticks(range(len(coeffs)), df.columns[:-1])  # Utilise les noms de colonnes du DataFrame
    plt.grid(True)
    
    # Afficher les prédictions vs réalité
    plt.subplot(2, 2, 3)
    predictions = np.dot(ga.X, ga.best_solution)
    plt.scatter(ga.Y, predictions, alpha=0.5)
    plt.plot([min(ga.Y), max(ga.Y)], [min(ga.Y), max(ga.Y)], 'r--')
    plt.xlabel("Ventes réelles")
    plt.ylabel("Ventes prédites")
    plt.title("Comparaison des prédictions et des valeurs réelles")
    plt.grid(True)
    
    # Résidus
    plt.subplot(2, 2, 4)
    residuals = ga.Y - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel("Prédictions")
    plt.ylabel("Résidus")
    plt.title("Graphique des résidus")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def split_data(df, test_size=0.2, random_state=42):
    # Mélanger les données
    df_shuffled = df.sample(frac=1, random_state=random_state)
    
    # Diviser les données
    test_size = int(len(df_shuffled) * test_size)
    train_df = df_shuffled[test_size:]
    test_df = df_shuffled[:test_size]
    
    return train_df, test_df

def main():
    # Vérifier que le DataFrame contient les données attendues
    expected_columns = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)', 'Sales ($)']
    
    # Vérifier les colonnes et les renommer si nécessaire
    if set(df.columns) != set(expected_columns):
        print(f"Attention: Les colonnes du fichier CSV ne correspondent pas aux colonnes attendues.")
        print(f"Colonnes attendues: {expected_columns}")
        print(f"Colonnes trouvées: {list(df.columns)}")
        
        # Si nous avons le bon nombre de colonnes mais avec des noms différents,
        # on peut supposer que l'ordre est le même et les renommer
        if len(df.columns) == len(expected_columns):
            df.columns = expected_columns
            print("Les colonnes ont été renommées selon le format attendu.")
    
    # Séparation des données en entraînement et test
    train_df, test_df = split_data(df)
    
    # Préparation des caractéristiques X et de la cible Y
    X_train = train_df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y_train = train_df.iloc[:, -1].values   # Dernière colonne (Sales)
    
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    
    # Normalisation des données pour de meilleurs résultats
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train_normalized = (X_train - X_mean) / X_std
    X_test_normalized = (X_test - X_mean) / X_std
    
    # Création et exécution de l'algorithme génétique
    ga = GeneticAlgorithm(X_train_normalized, y_train)
    best_solution, _, _ = ga.run()
    
    # Solution dénormalisée pour interprétation
    print("\nMeilleure solution trouvée (coefficients normalisés):")
    print(best_solution)
    
    # Évaluation sur l'ensemble de test
    test_predictions = np.dot(X_test_normalized, best_solution)
    test_error = np.mean(np.abs(test_predictions - y_test))
    print(f"Erreur sur l'ensemble de test: {test_error:.4f}")
    
    # Calcul du R²
    y_mean = np.mean(y_test)
    ss_total = np.sum((y_test - y_mean) ** 2)
    ss_residual = np.sum((y_test - test_predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R² sur l'ensemble de test: {r_squared:.4f}")
    
    # Affichage des résultats
    plot_results(ga)
    
    return best_solution, r_squared, test_error

if __name__ == "__main__":
    best_solution, r_squared, test_error = main()