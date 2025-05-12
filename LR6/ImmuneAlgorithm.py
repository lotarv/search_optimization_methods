import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Функция Розенброка для оптимизации
def rosenbrock_function(x, y):
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

class AIS:
    def __init__(self, iter_number, num_antibodies, num_best, num_clones, num_random, mutation_rate, x_range, y_range):
        self.iter_number = iter_number
        self.num_antibodies = num_antibodies
        self.num_best = num_best
        self.num_clones = num_clones
        self.num_random = num_random
        self.mutation_rate = mutation_rate
        self.x_range = x_range
        self.y_range = y_range
        self.delta_mutation = mutation_rate / iter_number if iter_number > 0 else 0

        # Инициализация популяции
        self.antibodies = np.random.uniform(low=[x_range[0], y_range[0]],
                                            high=[x_range[1], y_range[1]],
                                            size=(num_antibodies, 2))
        self.fitness = np.array([rosenbrock_function(ab[0], ab[1]) for ab in self.antibodies])
        self.antibody_best = self.antibodies[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)
        self.history = []

    def mutate(self, antibody):
        mutation = self.mutation_rate * np.random.uniform(-0.5, 0.5, 2)
        new_ab = np.clip(antibody + mutation, [self.x_range[0], self.y_range[0]], [self.x_range[1], self.y_range[1]])
        return new_ab

    def next_iteration(self):
        # Сортировка антител по фитнесу
        indices = np.argsort(self.fitness)
        self.antibodies = self.antibodies[indices]
        self.fitness = self.fitness[indices]

        # Выбор лучших антител
        best_antibodies = self.antibodies[:self.num_best]

        # Клонирование
        clones = np.repeat(best_antibodies, self.num_clones, axis=0)

        # Мутация клонов
        clones = np.array([self.mutate(clone) for clone in clones])
        clones_fitness = np.array([rosenbrock_function(ab[0], ab[1]) for ab in clones])

        # Выбор случайных клонов
        clone_indices = np.argsort(clones_fitness)[:self.num_random]
        clones = clones[clone_indices]
        clones_fitness = clones_fitness[clone_indices]

        # Объединение и сжатие
        combined = np.vstack((self.antibodies, clones))
        combined_fitness = np.concatenate((self.fitness, clones_fitness))
        combined_indices = np.argsort(combined_fitness)[:self.num_antibodies]
        self.antibodies = combined[combined_indices]
        self.fitness = combined_fitness[combined_indices]

        # Обновление лучшего решения
        if self.fitness[0] < self.best_fitness:
            self.best_fitness = self.fitness[0]
            self.antibody_best = self.antibodies[0]

        # Уменьшение mutation_rate
        self.mutation_rate -= self.delta_mutation

def ImmuneAlgorithm(frame, root, ax, canvas):
    def run_optimization():
        iter_number = iterations_var.get()
        antibodies_num = antibodies_number_var.get()
        best_num = best_number_var.get()
        random_num = random_number_var.get()
        clones_num = clones_number_var.get()
        mutation_coef = mutation_rate_var.get()
        delay = delay_var.get()
        z_min = z_interval_min.get()
        z_max = z_interval_max.get()

        # Проверка параметров
        if best_num > antibodies_num or random_num > best_num * clones_num:
            results_text.config(state=tk.NORMAL)
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, "Ошибка: Проверьте параметры (лучшие антитела <= антитела, выбираемые клоны <= клоны).\n")
            results_text.config(state=tk.DISABLED)
            return

        # Генерация сетки для графика
        x_range = np.linspace(x_interval_min.get(), x_interval_max.get(), 100)
        y_range = np.linspace(y_interval_min.get(), y_interval_max.get(), 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rosenbrock_function(X[i, j], Y[i, j])
        
        # Маскируем Z-значения, которые выходят за пределы z_max
        Z_masked = np.ma.masked_where(Z > z_max, Z)
        
        # Очистка и настройка графика
        ax.cla()
        ax.plot_surface(X, Y, Z_masked, cmap='viridis', alpha=0.7, vmin=z_min, vmax=z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim(z_min, z_max)
        ax.set_title("Иммунный алгоритм")

        ais = AIS(iter_number, antibodies_num, best_num, clones_num, random_num, mutation_coef,
                  [x_interval_min.get(), x_interval_max.get()], [y_interval_min.get(), y_interval_max.get()])

        # Отрисовка начальной популяции
        for ab in ais.antibodies:
            z_val = rosenbrock_function(ab[0], ab[1])
            if z_val <= z_max:
                ax.scatter(ab[0], ab[1], min(z_val, z_max), c="red", s=10)
        canvas.draw()
        root.update()
        time.sleep(delay)

        # Запуск итераций
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        cnt = 0
        prev_best = float('inf')
        for i in range(iter_number):
            ais.next_iteration()
            current_best = ais.best_fitness
            ais.history.append({
                'iteration': i + 1,
                'x': ais.antibody_best[0],
                'y': ais.antibody_best[1],
                'f_value': ais.best_fitness
            })
            if abs(prev_best - current_best) < 0.0001:
                cnt += 1
            else:
                cnt = 0
            if cnt == 15:
                break
            prev_best = current_best

            # Отрисовка текущей популяции
            ax.cla()
            ax.plot_surface(X, Y, Z_masked, cmap='viridis', alpha=0.7, vmin=z_min, vmax=z_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_zlim(z_min, z_max)
            ax.set_title("Иммунный алгоритм")
            for ab in ais.antibodies:
                z_val = rosenbrock_function(ab[0], ab[1])
                if z_val <= z_max:
                    ax.scatter(ab[0], ab[1], min(z_val, z_max), c="red", s=10)
            results_text.insert(tk.END,
                                f"Шаг {i + 1}: ({ais.antibody_best[0]:.4f}, {ais.antibody_best[1]:.4f}), f = {ais.best_fitness:.4f}\n")
            results_text.yview_moveto(1)
            canvas.draw()
            root.update()
            time.sleep(delay)

        # Финальный результат
        ax.cla()
        ax.plot_surface(X, Y, Z_masked, cmap='viridis', alpha=0.7, vmin=z_min, vmax=z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim(z_min, z_max)
        ax.set_title("Иммунный алгоритм")
        for ab in ais.antibodies:
            z_val = rosenbrock_function(ab[0], ab[1])
            if z_val <= z_max:
                ax.scatter(ab[0], ab[1], min(z_val, z_max), c="red", s=10)
        ax.scatter(ais.antibody_best[0], ais.antibody_best[1], min(ais.best_fitness, z_max), c='black', marker='x', s=60)
        canvas.draw()
        results_text.insert(tk.END,
                            f"Результат: ({ais.antibody_best[0]:.5f}, {ais.antibody_best[1]:.5f}), f = {ais.best_fitness:.8f}\n")
        results_text.yview_moveto(1)
        results_text.config(state=tk.DISABLED)

    # Интерфейс
    param_frame = frame
    ttk.Label(param_frame, text="Параметры иммунного алгоритма").grid(row=0, column=0, pady=10)
    ttk.Label(param_frame, text="Итерации").grid(row=1, column=0)
    ttk.Label(param_frame, text="Антитела").grid(row=2, column=0)
    ttk.Label(param_frame, text="Лучшие антитела").grid(row=3, column=0)
    ttk.Label(param_frame, text="Выбираемые клоны").grid(row=4, column=0)
    ttk.Label(param_frame, text="Клоны").grid(row=5, column=0)
    ttk.Label(param_frame, text="Мутации").grid(row=6, column=0)
    ttk.Label(param_frame, text="Задержка").grid(row=7, column=0)

    iterations_var = tk.IntVar(value=200)
    antibodies_number_var = tk.IntVar(value=50)
    best_number_var = tk.IntVar(value=10)
    random_number_var = tk.IntVar(value=10)
    clones_number_var = tk.IntVar(value=20)
    mutation_rate_var = tk.DoubleVar(value=0.2)
    delay_var = tk.DoubleVar(value=0.01)

    ttk.Entry(param_frame, textvariable=iterations_var).grid(row=1, column=1)
    ttk.Entry(param_frame, textvariable=antibodies_number_var).grid(row=2, column=1)
    ttk.Entry(param_frame, textvariable=best_number_var).grid(row=3, column=1)
    ttk.Entry(param_frame, textvariable=random_number_var).grid(row=4, column=1)
    ttk.Entry(param_frame, textvariable=clones_number_var).grid(row=5, column=1)
    ttk.Entry(param_frame, textvariable=mutation_rate_var).grid(row=6, column=1)
    ttk.Entry(param_frame, textvariable=delay_var).grid(row=7, column=1)

    ttk.Label(param_frame, text="Интервалы").grid(row=8, column=0, pady=10)
    ttk.Label(param_frame, text="X min").grid(row=9, column=0)
    ttk.Label(param_frame, text="X max").grid(row=10, column=0)
    ttk.Label(param_frame, text="Y min").grid(row=11, column=0)
    ttk.Label(param_frame, text="Y max").grid(row=12, column=0)
    ttk.Label(param_frame, text="Z min").grid(row=13, column=0)
    ttk.Label(param_frame, text="Z max").grid(row=14, column=0)

    x_interval_min = tk.DoubleVar(value=-5)
    x_interval_max = tk.DoubleVar(value=5)
    y_interval_min = tk.DoubleVar(value=-5)
    y_interval_max = tk.DoubleVar(value=5)
    z_interval_min = tk.DoubleVar(value=0)
    z_interval_max = tk.DoubleVar(value=80000)

    ttk.Entry(param_frame, textvariable=x_interval_min).grid(row=9, column=1)
    ttk.Entry(param_frame, textvariable=x_interval_max).grid(row=10, column=1)
    ttk.Entry(param_frame, textvariable=y_interval_min).grid(row=11, column=1)
    ttk.Entry(param_frame, textvariable=y_interval_max).grid(row=12, column=1)
    ttk.Entry(param_frame, textvariable=z_interval_min).grid(row=13, column=1)
    ttk.Entry(param_frame, textvariable=z_interval_max).grid(row=14, column=1)

    ttk.Button(param_frame, text="Запустить", command=run_optimization).grid(row=15, column=1, pady=10)
    results_text = scrolledtext.ScrolledText(param_frame, height=10, width=40, state=tk.DISABLED)
    results_text.grid(row=16, column=0, columnspan=2)