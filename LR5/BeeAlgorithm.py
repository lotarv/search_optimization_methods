import numpy as np
import random
from LR5.Bee import Bee
import tkinter as tk


class BeeAlgorithm:
    def __init__(self, num_scouts, elite_radius, perspective_radius, num_elite, num_perspective, agents_per_perspective,
                 agents_per_elite, bounds, max_epochs, stagnation_limit, fitness_function):
        '''
        - num_scouts (int): Количество пчел-разведчиков в популяции.
        - elite_radius (float): Радиус элитных участков для каждой пчелы.
        - perspective_radius (float): Радиус перспективных участков для каждой пчелы.
        - num_elite (int): Количество элитных участков для обновления координат пчелы.
        - num_perspective (int): Количество перспективных участков для обновления координат пчелы.
        - agents_per_perspective (int): Количество агентов, отправляемых на каждый перспективный участок.
        - agents_per_elite (int): Количество агентов, отправляемых на каждый элитный участок.
        - bounds (list): Границы пространства поиска. Список кортежей, где каждый кортеж - границы для соответствующего измерения.
        - max_epochs (int): Максимальное количество итераций алгоритма.
        - stagnation_limit (int): Количество итераций, после которого процесс считается застойным (условие останова).
        - fitness_function: заданная фитнесс-функция
        '''
        self.num_scouts = num_scouts
        self.initial_elite_radius = elite_radius
        self.initial_perspective_radius = perspective_radius
        self.num_elite = num_elite
        self.num_perspective = num_perspective
        self.agents_per_perspective = agents_per_perspective
        self.agents_per_elite = agents_per_elite
        self.bounds = bounds
        self.max_epochs = max_epochs
        self.stagnation_limit = stagnation_limit
        self.best_bees = []
        self.fitness_function = fitness_function
        self.current_epoch = 0
        self.shake_count = 0
        self.max_shakes = 3

    def set_options(self, root, ax, canvas, results_text, bound_start, bound_end, target_func):
        self.canvas = canvas
        self.root = root
        self.ax = ax
        self.results_text = results_text
        self.bound_start = bound_start
        self.bound_end = bound_end
        self.target_func = target_func

    def initialize_bees(self):
        bees = []
        for _ in range(self.num_scouts):
            coords = np.array([random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))], dtype='float')
            bees.append(Bee(coords, self.fitness_function(coords)))
        return bees

    def optimize(self):
        bees = self.initialize_bees()
        stagnation_count = 0
        best_fitness = float('inf')

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            bees = sorted(bees, key=lambda bee: bee.fitness)
            self.best_bees = bees[:self.num_elite]

            x_range = np.linspace(self.bound_start, self.bound_end, 100)
            y_range = np.linspace(self.bound_start, self.bound_end, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.target_func(np.array([X[i, j], Y[i, j]]))

            self.ax.cla()
            self.canvas.draw()
            self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_xticks(np.arange(self.bound_start, self.bound_end + 1, 2))
            self.ax.set_yticks(np.arange(self.bound_start, self.bound_end + 1, 2))

            for i in range(self.num_scouts):
                self.explore(bees[i])
                self.ax.scatter(bees[i].coords[0], bees[i].coords[1], bees[i].fitness, color='black', s=10)

            bees = self.select_best(bees)

            current_best_fitness = self.best_bees[0].fitness
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= self.stagnation_limit:
                if self.shake_count < self.max_shakes:
                    self.results_text.insert(tk.END, f"\nСтагнация. Выполняется встряска {self.shake_count + 1} на итерации {epoch}.\n")
                    bees = bees[:self.num_perspective] + self.initialize_bees()[:self.num_scouts - self.num_perspective]
                    self.shake_count += 1
                    stagnation_count = 0  # Сбрасываем счетчик стагнации после встряски
                else:
                    self.results_text.insert(tk.END, f"\nСтагнация. Достигнуто максимальное количество встрясок ({self.max_shakes}). Оптимизация остановлена на итерации {epoch}.\n")
                    break

            self.results_text.insert(tk.END,
                                    f"Итерация {epoch}: Лучшее решение ({self.best_bees[0].coords[0]:.8f}, {self.best_bees[0].coords[1]:.8f}, {self.best_bees[0].fitness:.8f})\n")
            self.canvas.draw()
            self.results_text.yview_moveto(1)
            self.root.update()

        self.best_bees = sorted(bees, key=lambda bee: bee.fitness)[:self.num_elite]
        return self.best_bees[0]

    def explore(self, bee):
        # Адаптивное уменьшение радиусов
        shrink_factor = 1 - (self.current_epoch / self.max_epochs)  # Уменьшаем радиус с течением времени
        elite_radius = self.initial_elite_radius * shrink_factor
        perspective_radius = self.initial_perspective_radius * shrink_factor

        # Упрощенная формула: комбинация движения к лучшей пчеле и случайного исследования
        phi = random.uniform(-0.5, 0.5)  # Уменьшенный диапазон для более плавного движения
        new_coords = np.copy(bee.coords)

        for i in range(len(bee.coords)):
            # Движение к лучшей пчеле с небольшим случайным возмущением
            best_bee = self.best_bees[0]
            new_coords[i] += phi * (best_bee.coords[i] - bee.coords[i])
            # Добавляем случайное исследование в пределах радиусов
            new_coords[i] += random.uniform(-perspective_radius, perspective_radius)
            # Если пчела элитная, используем меньший радиус
            if bee in self.best_bees:
                new_coords[i] += random.uniform(-elite_radius, elite_radius)

        # Ограничение координат
        new_coords = np.array([max(min(new_coords[i], self.bounds[i][1]), self.bounds[i][0]) for i in range(len(new_coords))])

        # Обновление, если новое решение лучше
        new_fitness = self.fitness_function(new_coords)
        if new_fitness < bee.fitness:
            bee.coords = new_coords
            bee.fitness = new_fitness

    def select_best(self, bees):
        # Сохраняем больше перспективных пчел
        bees.sort(key=lambda bee: bee.fitness)
        keep_ratio = 0.7  # Сохраняем 70% лучших пчел
        keep_count = int(self.num_scouts * keep_ratio)
        return bees[:keep_count] + self.initialize_bees()[:self.num_scouts - keep_count]