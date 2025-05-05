import numpy as np
import random
import tkinter as tk


class Bee:
    def __init__(self, coords, fitness, bounds, fitness_function):
        self.coords = np.array(coords, dtype=float)
        self.fitness = fitness
        self.bounds = bounds
        self.fitness_function = fitness_function

    def calcFitness(self):
        self.fitness = self.fitness_function(self.coords)

    def getPosition(self):
        return self.coords.tolist()

    def goto(self, otherpos, radius):
        self.coords = np.array([otherpos[n] + random.uniform(-radius, radius) for n in range(len(otherpos))])
        self.checkPosition()
        self.calcFitness()

    def checkPosition(self):
        for i in range(len(self.coords)):
            self.coords[i] = max(min(self.coords[i], self.bounds[i][1]), self.bounds[i][0])

    def otherPatch(self, bee_list, radius):
        if not bee_list:
            return True
        self.calcFitness()
        for bee in bee_list:
            bee.calcFitness()
            pos = bee.getPosition()
            for i in range(len(self.coords)):
                if abs(self.coords[i] - pos[i]) > radius:
                    return True
        return False


class BeeAlgorithm:
    def __init__(self, num_scouts, elite_radius, perspective_radius, num_elite, num_perspective, agents_per_perspective,
                 agents_per_elite, bounds, max_epochs, stagnation_limit, fitness_function):
        '''
        - num_scouts (int): Количество пчел-разведчиков в популяции.
        - elite_radius (float): Радиус элитных участков.
        - perspective_radius (float): Радиус перспективных участков.
        - num_elite (int): Количество элитных участков.
        - num_perspective (int): Количество перспективных участков.
        - agents_per_perspective (int): Количество пчел для перспективных участков.
        - agents_per_elite (int): Количество пчел для элитных участков.
        - bounds (list): Границы поиска [(x_min, x_max), (y_min, y_max)].
        - max_epochs (int): Максимальное количество итераций.
        - stagnation_limit (int): Итераций без улучшения для стагнации.
        - fitness_function: Функция для оптимизации.
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
        self.fitness_function = fitness_function
        self.best_bees = []
        self.bestsites = []
        self.selectedsites = []
        self.current_epoch = 0
        self.shake_count = 0
        self.max_shakes = 3
        self.best_fitness = float('inf')

    def set_options(self, root, ax, canvas, results_text, bound_start, bound_end, target_func):
        self.canvas = canvas
        self.root = root
        self.ax = ax
        self.results_text = results_text
        self.bound_start = bound_start
        self.bound_end = bound_end
        self.target_func = target_func

    def initialize_bees(self):
        # Создаем популяцию, учитывая разведчиков, элитные и перспективные пчелы
        bee_count = self.num_scouts + self.agents_per_perspective * self.num_perspective + self.agents_per_elite * self.num_elite
        bees = []
        for _ in range(bee_count):
            coords = np.array([random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))], dtype=float)
            fitness = self.fitness_function(coords)
            bees.append(Bee(coords, fitness, self.bounds, self.fitness_function))
        return bees

    def explore(self, bee, elite_radius, perspective_radius):
        phi = random.uniform(-0.5, 0.5)
        new_coords = np.copy(bee.coords)

        for i in range(len(bee.coords)):
            best_bee = self.best_bees[0]
            new_coords[i] += phi * (best_bee.coords[i] - bee.coords[i])
            new_coords[i] += random.uniform(-perspective_radius, perspective_radius)
            if bee in self.best_bees:
                new_coords[i] += random.uniform(-elite_radius, elite_radius)

        new_coords = np.array([max(min(new_coords[i], self.bounds[i][1]), self.bounds[i][0]) for i in range(len(new_coords))])
        new_fitness = self.fitness_function(new_coords)
        if new_fitness < bee.fitness:
            bee.coords = new_coords
            bee.fitness = new_fitness

    def sendBees(self, position, index, count, bees, radius):
        for i in range(count):
            if index >= len(bees):
                break
            bee = bees[index]
            if bee not in self.bestsites and bee not in self.selectedsites:
                bee.goto(position, radius)
            index += 1
        return index

    def select_best(self, bees):
        bees.sort(key=lambda x: x.fitness)
        keep_ratio = 0.7
        keep_count = int(len(bees) * keep_ratio)
        return bees[:keep_count] + self.initialize_bees()[:len(bees) - keep_count]

    def optimize(self):
        bees = self.initialize_bees()
        stagnation_count = 0
        message = "Достигнуто максимальное количество итераций"

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Адаптивное уменьшение радиусов
            shrink_factor = 1 - (self.current_epoch / self.max_epochs)
            elite_radius = self.initial_elite_radius * shrink_factor
            perspective_radius = self.initial_perspective_radius * shrink_factor

            # Сортировка пчел и выбор лучших
            for bee in bees:
                bee.calcFitness()
            bees.sort(key=lambda x: x.fitness)
            self.best_bees = bees[:self.num_elite]

            # Визуализация
            x_range = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
            y_range = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
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
            self.ax.set_xticks(np.arange(self.bounds[0][0], self.bounds[0][1] + 1, 2))
            self.ax.set_yticks(np.arange(self.bounds[1][0], self.bounds[1][1] + 1, 2))

            # Выбор элитных и перспективных участков
            self.bestsites = [bees[0]]
            self.selectedsites = []
            curr_index = 1
            while curr_index < len(bees) and len(self.bestsites) < self.num_elite:
                bee = bees[curr_index]
                if bee.otherPatch(self.bestsites, elite_radius):
                    self.bestsites.append(bee)
                curr_index += 1

            while curr_index < len(bees) and len(self.selectedsites) < self.num_perspective:
                bee = bees[curr_index]
                if bee.otherPatch(self.bestsites, perspective_radius) and \
                   bee.otherPatch(self.selectedsites, perspective_radius):
                    self.selectedsites.append(bee)
                curr_index += 1

            # Отправка пчел в участки
            bee_index = 1
            for best_bee in self.bestsites:
                bee_index = self.sendBees(best_bee.getPosition(), bee_index, self.agents_per_elite, bees, elite_radius)
            for sel_bee in self.selectedsites:
                bee_index = self.sendBees(sel_bee.getPosition(), bee_index, self.agents_per_perspective, bees, perspective_radius)

            # Остальные пчелы используют explore
            for i in range(bee_index, len(bees)):
                self.explore(bees[i], elite_radius, perspective_radius)

            # Визуализация всех пчел
            for bee in bees:
                self.ax.scatter(bee.coords[0], bee.coords[1], bee.fitness, color='black', s=10)

            # Сохранение лучших
            bees = self.select_best(bees)

            # Проверка стагнации и встряска
            current_best_fitness = self.best_bees[0].fitness
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= self.stagnation_limit:
                if self.shake_count < self.max_shakes:
                    self.results_text.insert(tk.END, f"\nСтагнация. Выполняется встряска {self.shake_count + 1} на итерации {epoch}.\n")
                    bees.sort(key=lambda x: x.fitness)
                    bees = bees[:self.num_elite] + self.initialize_bees()[:len(bees) - self.num_elite]
                    self.shake_count += 1
                    stagnation_count = 0
                else:
                    self.results_text.insert(tk.END, f"\nСтагнация. Достигнуто максимальное количество встрясок ({self.max_shakes}). Оптимизация остановлена на итерации {epoch}.\n")
                    break

            self.results_text.insert(tk.END,
                                    f"Итерация {epoch}: Лучшее решение ({self.best_bees[0].coords[0]:.8f}, {self.best_bees[0].coords[1]:.8f}, {self.best_bees[0].fitness:.8f})\n")
            self.canvas.draw()
            self.results_text.yview_moveto(1)
            self.root.update()

        # Финальная сортировка и вывод лучшего решения
        self.best_bees = sorted(bees, key=lambda x: x.fitness)[:self.num_elite]
        self.results_text.insert(tk.END,
                                f"Лучшее решение ({self.best_bees[0].coords[0]:.8f}, {self.best_bees[0].coords[1]:.8f}, {self.best_bees[0].fitness:.8f})\n")
        self.ax.scatter(self.best_bees[0].coords[0], self.best_bees[0].coords[1], self.best_bees[0].fitness, c="red")
        self.canvas.draw()
        return self.best_bees[0]