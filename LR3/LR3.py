import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np

def GeneticAlgorithm(frame, root, ax, canvas):
    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def rastrigin(x, y):
        return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

    def selection(population, fitness_scores):
        probabilities = 1 / (fitness_scores + 1e-8)
        probabilities /= np.sum(probabilities)
        selected_indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
        return [population[i] for i in selected_indices]

    def crossover(parent1, parent2, rate, swap_prob=0.5):
        if np.random.rand() < rate:
            child = np.zeros_like(parent1)
            for i in range(len(parent1)):
                if np.random.rand() < swap_prob:
                    child[i] = parent2[i]  # Берем ген от parent2
                else:
                    child[i] = parent1[i]  # Берем ген от parent1
        else:
            child = parent1.copy()
        return child

    def mutate(individual, rate, bounds):
        for i in range(len(individual)):
            if np.random.rand() < rate:
                individual[i] += np.random.uniform(-0.5, 0.5)
                individual[i] = np.clip(individual[i], bounds[i][0], bounds[i][1])
        return individual

    def run_optimization():
        # Выбор функции
        selected_func = func_var.get()
        if selected_func == "Rosenbrock":
            target_function = rosenbrock
        else:
            target_function = rastrigin

        x_range = np.linspace(x_interval_min.get(), x_interval_max.get(), 100)
        y_range = np.linspace(y_interval_min.get(), y_interval_max.get(), 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = target_function(X, Y)

        pop_size = int(x_var.get())
        generations = int(y_var.get())
        mutation_rate = float(mutation_var.get())
        crossover_rate = float(crossover_var.get())
        stagnation_limit = int(stagnation_var.get())

        bounds = [(x_interval_min.get(), x_interval_max.get()), (y_interval_min.get(), y_interval_max.get())]
        population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, 2))

        best_global = None
        best_global_fitness = float('inf')
        stagnation_counter = 0

        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)

        for generation in range(generations):
            fitness_scores = np.array([target_function(x, y) for x, y in population])
            best_idx = np.argmin(fitness_scores)
            best_individual = population[best_idx]
            best_fitness = fitness_scores[best_idx]

            if best_fitness < best_global_fitness - 1e-8:
                best_global_fitness = best_fitness
                best_global = best_individual.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_limit:
                results_text.insert(tk.END, f"Остановка из-за стагнации на поколении {generation}\n")
                break

            new_population = [best_individual]  # Элитизм
            while len(new_population) < pop_size:
                parent1, parent2 = selection(population, fitness_scores)
                child = crossover(parent1, parent2, crossover_rate)
                child = mutate(child, mutation_rate, bounds)
                new_population.append(child)

            population = np.array(new_population)

            ax.cla()
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xticks(np.arange(bounds[0][0], bounds[0][1] + 1, x_axis_interval.get()))
            ax.set_yticks(np.arange(bounds[1][0], bounds[1][1] + 1, y_axis_interval.get()))
            ax.set_title(f"Генетический алгоритм: {selected_func}")

            for i in range(len(fitness_scores)):
                ax.scatter(population[i][0], population[i][1], fitness_scores[i], color='red', s=10)

            ax.scatter(best_global[0], best_global[1], best_global_fitness, color='black', marker='x', s=60)
            canvas.draw()
            root.update()

            results_text.insert(tk.END,
                f"Поколение {generation}: Лучшее ({best_individual[0]:.4f}, {best_individual[1]:.4f}), "
                f"f(x,y) = {best_fitness:.7f}\n")
            results_text.yview_moveto(1)

        results_text.insert(tk.END,
            f"\nЗавершено. Глобальный минимум: ({best_global[0]:.6f}, {best_global[1]:.6f}), "
            f"f(x,y) = {best_global_fitness:.10f}\n")
        results_text.config(state=tk.DISABLED)

    # --- UI ---
    ttk.Label(frame, text="Инициализация значений", font=("Helvetica", 12)).grid(row=0, column=0, pady=15)
    ttk.Label(frame, text="Функция").grid(row=1, column=0)
    ttk.Label(frame, text="Особи").grid(row=2, column=0)
    ttk.Label(frame, text="Итерации").grid(row=3, column=0)
    ttk.Label(frame, text="Mutation rate").grid(row=4, column=0)
    ttk.Label(frame, text="Crossover rate").grid(row=5, column=0)
    ttk.Label(frame, text="Стагнация (поколений)").grid(row=6, column=0)

    func_var = tk.StringVar(value="Rosenbrock")
    func_menu = ttk.Combobox(frame, textvariable=func_var, values=["Rosenbrock", "Rastrigin"], state="readonly")
    func_menu.grid(row=1, column=1)

    x_var = tk.DoubleVar(value=100)
    y_var = tk.DoubleVar(value=100)
    mutation_var = tk.DoubleVar(value=0.1)
    crossover_var = tk.DoubleVar(value=0.9)
    stagnation_var = tk.IntVar(value=5)

    ttk.Entry(frame, textvariable=x_var).grid(row=2, column=1)
    ttk.Entry(frame, textvariable=y_var).grid(row=3, column=1)
    ttk.Entry(frame, textvariable=mutation_var).grid(row=4, column=1)
    ttk.Entry(frame, textvariable=crossover_var).grid(row=5, column=1)
    ttk.Entry(frame, textvariable=stagnation_var).grid(row=6, column=1)

    ttk.Label(frame, text="X интервал (min)").grid(row=7, column=0)
    ttk.Label(frame, text="X интервал (max)").grid(row=8, column=0)
    ttk.Label(frame, text="Y интервал (min)").grid(row=9, column=0)
    ttk.Label(frame, text="Y интервал (max)").grid(row=10, column=0)
    ttk.Label(frame, text="Ось X интервал").grid(row=11, column=0)
    ttk.Label(frame, text="Ось Y интервал").grid(row=12, column=0)

    x_interval_min = tk.DoubleVar(value=-5)
    x_interval_max = tk.DoubleVar(value=5)
    y_interval_min = tk.DoubleVar(value=-5)
    y_interval_max = tk.DoubleVar(value=5)
    x_axis_interval = tk.IntVar(value=2)
    y_axis_interval = tk.IntVar(value=2)

    ttk.Entry(frame, textvariable=x_interval_min).grid(row=7, column=1)
    ttk.Entry(frame, textvariable=x_interval_max).grid(row=8, column=1)
    ttk.Entry(frame, textvariable=y_interval_min).grid(row=9, column=1)
    ttk.Entry(frame, textvariable=y_interval_max).grid(row=10, column=1)
    ttk.Entry(frame, textvariable=x_axis_interval).grid(row=11, column=1)
    ttk.Entry(frame, textvariable=y_axis_interval).grid(row=12, column=1)

    ttk.Button(frame, text="Выполнить", command=run_optimization).grid(row=13, column=1, pady=10)
    results_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=18, width=40, state=tk.DISABLED)
    results_text.grid(row=14, column=0, columnspan=2, pady=10)
