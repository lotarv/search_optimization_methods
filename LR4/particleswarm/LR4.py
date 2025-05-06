import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import math
from LR4.particleswarm.swarm import Swarm
import numpy


class Swarm_Rastrigin(Swarm):
    def __init__(self, swarmsize, minvalues, maxvalues, currentVelocityRatio, localVelocityRatio, globalVelocityRatio):
        super().__init__(swarmsize, minvalues, maxvalues, currentVelocityRatio, localVelocityRatio, globalVelocityRatio)

    def _finalFunc(self, position):
        function = 10.0 * len(self.minvalues) + sum(position * position - 10.0 * numpy.cos(2 * numpy.pi * position))
        penalty = self._getPenalty(position, 10000.0)
        return function + penalty
import numpy
from LR4.particleswarm.swarm import Swarm


class Swarm_Rosenbrock(Swarm):
    def __init__(self, swarmsize, minvalues, maxvalues, currentVelocityRatio, localVelocityRatio, globalVelocityRatio):
        super().__init__(swarmsize, minvalues, maxvalues, currentVelocityRatio, localVelocityRatio, globalVelocityRatio)

    def _finalFunc(self, position):
        function = sum([100 * (position[i + 1] - position[i] ** 2) ** 2 + (1 - position[i]) ** 2
                        for i in range(len(position) - 1)])
        penalty = self._getPenalty(position, 10000.0)
        return function + penalty

def printResult(swarm, iteration):
    template = u""" Лучшие координаты: {bestpos}\n Лучший результат: {finalfunc}\n\n"""
    return template.format(iter=iteration, bestpos=swarm.globalBestPosition, finalfunc=swarm.globalBestFinalFunc)


def ParticleSwarmAlgorithm(frame, root, ax, canvas):
    def rastrigin(*X):
        A = 10
        return A * len(X) + sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])

    def rosenbrock(*X):
        return sum([100 * (X[i + 1] - X[i] ** 2) ** 2 + (1 - X[i]) ** 2 for i in range(len(X) - 1)])

    def run_optimization():
        X = np.linspace(x_interval_min.get(), x_interval_max.get(), 200)
        Y = np.linspace(y_interval_min.get(), y_interval_max.get(), 200)
        X, Y = np.meshgrid(X, Y)

        selected_function = function_var.get()
        if selected_function == "Функция Растригина":
            target_function = rastrigin
            swarm_class = Swarm_Rastrigin
        elif selected_function == "Функция Розенброка":
            target_function = rosenbrock
            swarm_class = Swarm_Rosenbrock
        else:
            raise ValueError("Неизвестная функция")

        Z = np.array([[target_function(x, y) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        iterCount = iteration.get()
        dimension = 2
        swarmsize = particle.get()
        minvalues = np.array([-5.12] * dimension)
        maxvalues = np.array([5.12] * dimension)   

        currentVelocityRatio = inertia.get()
        localVelocityRatio = alpha.get()
        globalVelocityRatio = beta.get()

        swarm = swarm_class(swarmsize, minvalues, maxvalues, currentVelocityRatio, localVelocityRatio, globalVelocityRatio)

        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)

        best_value = float('inf')
        no_improvement_counter = 0

        for n in range(iterCount):
            ax.cla()
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xticks(np.arange(x_interval_min.get(), x_interval_max.get() + 1, x_axis_interval.get()))
            ax.set_yticks(np.arange(y_interval_min.get(), y_interval_max.get() + 1, y_axis_interval.get()))
            ax.set_title("Алгоритм Роя Частиц")

            for particle_instance in swarm:
                ax.scatter(
                    particle_instance.position[0],
                    particle_instance.position[1],
                    target_function(particle_instance.position[0], particle_instance.position[1]),
                    color='red',
                    s=10
                )

            best = swarm.globalBestPosition
            best_z = target_function(best[0], best[1])
            ax.scatter(best[0], best[1], best_z, color='gold', s=50, marker='*', label='Лучший')
            ax.legend()

            results_text.insert(tk.END, f"Итерация {n + 1}\n")
            results_text.insert(tk.END, f"Лучшая позиция: {swarm.globalBestPosition}\n")
            results_text.insert(tk.END, f"Лучшее значение: {swarm.globalBestFinalFunc:.4f}\n")
            results_text.insert(tk.END, "-" * 30 + "\n")
            results_text.yview_moveto(1)

            canvas.draw()
            root.update()
            swarm.nextIteration()

            current_best = swarm.globalBestFinalFunc
            if abs(current_best - best_value) < 1e-6:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0
                best_value = current_best

            if no_improvement_counter >= stagnation_limit.get():
                results_text.insert(tk.END, f"Остановка: стагнация на {stagnation_limit.get()} итерациях\n")
                break

        results_text.config(state=tk.DISABLED)

    param_frame2 = frame

    # Параметры задачи
    ttk.Label(param_frame2, text="Инициализация значений", font=("Helvetica", 12)).grid(row=0, column=0, pady=15)
    ttk.Label(param_frame2, text="Итераций", font=("Helvetica", 10)).grid(row=1, column=0)
    ttk.Label(param_frame2, text="Частиц", font=("Helvetica", 10)).grid(row=2, column=0)
    ttk.Label(param_frame2, text="Альфа", font=("Helvetica", 10)).grid(row=3, column=0)
    ttk.Label(param_frame2, text="Бета", font=("Helvetica", 10)).grid(row=4, column=0)
    ttk.Label(param_frame2, text="Инерция", font=("Helvetica", 10)).grid(row=5, column=0)
    ttk.Label(param_frame2, text="Стагнация (итераций)", font=("Helvetica", 10)).grid(row=6, column=0)

    iteration = tk.IntVar(value=100)
    particle = tk.IntVar(value=200)
    alpha = tk.IntVar(value=2)
    beta = tk.IntVar(value=5)
    inertia = tk.DoubleVar(value=0.5)
    stagnation_limit = tk.IntVar(value=10)

    iteration_entry = ttk.Entry(param_frame2, textvariable=iteration)
    particle_entry = ttk.Entry(param_frame2, textvariable=particle)
    alpha_entry = ttk.Entry(param_frame2, textvariable=alpha)
    beta_entry = ttk.Entry(param_frame2, textvariable=beta)
    inertia_entry = ttk.Entry(param_frame2, textvariable=inertia)
    stagnation_entry = ttk.Entry(param_frame2, textvariable=stagnation_limit)

    iteration_entry.grid(row=1, column=1)
    particle_entry.grid(row=2, column=1)
    alpha_entry.grid(row=3, column=1)
    beta_entry.grid(row=4, column=1)
    inertia_entry.grid(row=5, column=1)
    stagnation_entry.grid(row=6, column=1)

    separator = ttk.Separator(param_frame2, orient="horizontal")
    separator.grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

    # Параметры функции
    ttk.Label(param_frame2, text="Функция и отображение ее графика", font=("Helvetica", 12)).grid(row=8, column=0, pady=10)
    ttk.Label(param_frame2, text="Выберите функцию", font=("Helvetica", 10)).grid(row=9, column=0)
    function_choices = ["Функция Растригина", "Функция Розенброка"]
    function_var = tk.StringVar(value=function_choices[0])
    function_menu = ttk.Combobox(param_frame2, textvariable=function_var, values=function_choices, width=22)
    function_menu.grid(row=9, column=1, pady=5)

    ttk.Label(param_frame2, text="X интервал (min)", font=("Helvetica", 10)).grid(row=10, column=0)
    ttk.Label(param_frame2, text="X интервал (max)", font=("Helvetica", 10)).grid(row=11, column=0)
    ttk.Label(param_frame2, text="Y интервал (min)", font=("Helvetica", 10)).grid(row=12, column=0)
    ttk.Label(param_frame2, text="Y интервал (max)", font=("Helvetica", 10)).grid(row=13, column=0)
    ttk.Label(param_frame2, text="Ось X интервал", font=("Helvetica", 10)).grid(row=14, column=0)
    ttk.Label(param_frame2, text="Ось Y интервал", font=("Helvetica", 10)).grid(row=15, column=0)

    x_interval_min = tk.DoubleVar(value=-4)
    x_interval_max = tk.DoubleVar(value=4)
    y_interval_min = tk.DoubleVar(value=-4)
    y_interval_max = tk.DoubleVar(value=4)
    x_axis_interval = tk.IntVar(value=2)
    y_axis_interval = tk.IntVar(value=2)

    x_interval_min_entry = ttk.Entry(param_frame2, textvariable=x_interval_min)
    x_interval_max_entry = ttk.Entry(param_frame2, textvariable=x_interval_max)
    y_interval_min_entry = ttk.Entry(param_frame2, textvariable=y_interval_min)
    y_interval_max_entry = ttk.Entry(param_frame2, textvariable=y_interval_max)
    x_axis_interval_entry = ttk.Entry(param_frame2, textvariable=x_axis_interval)
    y_axis_interval_entry = ttk.Entry(param_frame2, textvariable=y_axis_interval)

    x_interval_min_entry.grid(row=10, column=1)
    x_interval_max_entry.grid(row=11, column=1)
    y_interval_min_entry.grid(row=12, column=1)
    y_interval_max_entry.grid(row=13, column=1)
    x_axis_interval_entry.grid(row=14, column=1)
    y_axis_interval_entry.grid(row=15, column=1)

    separator = ttk.Separator(param_frame2, orient="horizontal")
    separator.grid(row=16, column=0, columnspan=2, sticky="ew", pady=10)

    ttk.Label(param_frame2, text="Выполнение и результаты", font=("Helvetica", 12)).grid(row=17, column=0, pady=10)
    results_text = scrolledtext.ScrolledText(param_frame2, wrap=tk.WORD, height=18, width=40, padx=2, state=tk.DISABLED)
    results_text.grid(row=18, column=0, padx=10)

    button_style = ttk.Style()
    button_style.configure("My.TButton", font=("Helvetica", 14))
    apply_settings_button = ttk.Button(param_frame2, text="Выполнить", command=run_optimization, style="My.TButton")
    apply_settings_button.grid(row=18, column=1, padx=10, pady=10)