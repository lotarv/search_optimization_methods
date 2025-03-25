import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import numdifftools as nd
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def GradientDescentAlgorithm(frame, root, ax, canvas):
    # Основная функция f(x) = 2x₁² + x₁x₂ + x₂²
    def target_function(x, y):
        return 2 * x**2 + x * y + y**2

    # Градиент функции
    def gradient(x, y):
        return np.array([4 * x + y, x + 2 * y])

    # Функция для запуска оптимизации
    def run_optimization():
        # Получение параметров из интерфейса
        x0 = x_var.get()
        y0 = y_var.get()
        step = step_var.get()
        max_iterations = iterations_var.get()
        delay = delay_var.get()
        eps = eps_var.get()  # Точность для условия остановки

        # Очистка текущего графика
        ax.cla()

        # Генерация сетки для графика
        x_range = np.linspace(x_interval_min.get(), x_interval_max.get(), 100)
        y_range = np.linspace(y_interval_min.get(), y_interval_max.get(), 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = target_function(X, Y)

        # Построение поверхности
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.set_zlabel('f(x)')
        ax.set_xticks(np.arange(x_interval_min.get(), x_interval_max.get() + 1, x_axis_interval.get()))
        ax.set_yticks(np.arange(y_interval_min.get(), y_interval_max.get() + 1, y_axis_interval.get()))
        ax.set_title("Градиентный спуск: f(x₁, x₂) = 2x₁² + x₁x₂ + x₂²")

        # Инициализация результатов
        results = []
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)

        # Основной цикл градиентного спуска
        curr_point = [x0, y0]
        iteration_count = 0

        while iteration_count < max_iterations:
            grad = gradient(curr_point[0], curr_point[1])
            grad_norm = np.linalg.norm(grad)

            # Условие остановки по норме градиента
            if grad_norm < eps:
                break

            # Шаг градиентного спуска
            next_point = [curr_point[0] - step * grad[0], curr_point[1] - step * grad[1]]
            f_curr = target_function(curr_point[0], curr_point[1])
            f_next = target_function(next_point[0], next_point[1])

            # Уменьшение шага, если значение функции не уменьшилось
            current_step = step
            while f_next >= f_curr:
                current_step /= 2
                next_point = [curr_point[0] - current_step * grad[0], curr_point[1] - current_step * grad[1]]
                f_next = target_function(next_point[0], next_point[1])
                if current_step < 1e-10:  # Предотвращение бесконечного цикла
                    break

            # Условие остановки по изменению точки и функции
            dist = np.sqrt((next_point[0] - curr_point[0])**2 + (next_point[1] - curr_point[1])**2)
            if dist < eps and abs(f_next - f_curr) < eps:
                curr_point = next_point
                break

            # Обновление текущей точки
            curr_point = next_point
            iteration_count += 1

            # Сохранение результатов
            f_value = target_function(curr_point[0], curr_point[1])
            results.append((curr_point[0], curr_point[1], iteration_count, f_value))

            # Обновление графика и текста
            ax.scatter([curr_point[0]], [curr_point[1]], [f_value], color='red', s=10)
            results_text.insert(tk.END,
                               f"Шаг {iteration_count}: ({curr_point[0]:.4f}, {curr_point[1]:.4f}), f(x) = {f_value:.7f}\n")
            results_text.yview_moveto(1)
            canvas.draw()
            root.update()
            time.sleep(delay)

        # Вывод финального результата
        if results:
            last_result = results[-1]
            ax.scatter([last_result[0]], [last_result[1]], [last_result[3]], color='black', marker='x', s=60)
            results_text.insert(tk.END,
                               f"Результат:\nКоординаты ({last_result[0]:.8f}, {last_result[1]:.8f})\n"
                               f"Значение функции: {last_result[3]:.8f}\n"
                               f"Итераций: {iteration_count}\n")
        else:
            results_text.insert(tk.END, "Сходимость не достигнута\n")
        results_text.yview_moveto(1)
        results_text.config(state=tk.DISABLED)

    # Настройка интерфейса
    param_frame = frame

    # Параметры задачи
    ttk.Label(param_frame, text="Параметры градиентного спуска", font=("Helvetica", 12)).grid(row=0, column=0, pady=15)
    ttk.Label(param_frame, text="X₁ начальное", font=("Helvetica", 10)).grid(row=1, column=0)
    ttk.Label(param_frame, text="X₂ начальное", font=("Helvetica", 10)).grid(row=2, column=0)
    ttk.Label(param_frame, text="Шаг", font=("Helvetica", 10)).grid(row=3, column=0)
    ttk.Label(param_frame, text="Макс. итераций", font=("Helvetica", 10)).grid(row=4, column=0)
    ttk.Label(param_frame, text="Задержка (сек)", font=("Helvetica", 10)).grid(row=5, column=0)
    ttk.Label(param_frame, text="Точность (ε)", font=("Helvetica", 10)).grid(row=6, column=0)

    x_var = tk.DoubleVar(value=-1.0)
    y_var = tk.DoubleVar(value=-1.0)
    step_var = tk.DoubleVar(value=0.1)
    iterations_var = tk.IntVar(value=100)
    delay_var = tk.DoubleVar(value=0.5)
    eps_var = tk.DoubleVar(value=0.0001)

    ttk.Entry(param_frame, textvariable=x_var).grid(row=1, column=1)
    ttk.Entry(param_frame, textvariable=y_var).grid(row=2, column=1)
    ttk.Entry(param_frame, textvariable=step_var).grid(row=3, column=1)
    ttk.Entry(param_frame, textvariable=iterations_var).grid(row=4, column=1)
    ttk.Entry(param_frame, textvariable=delay_var).grid(row=5, column=1)
    ttk.Entry(param_frame, textvariable=eps_var).grid(row=6, column=1)

    ttk.Separator(param_frame, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

    # Параметры графика
    ttk.Label(param_frame, text="Параметры графика", font=("Helvetica", 12)).grid(row=8, column=0, pady=10)
    ttk.Label(param_frame, text="X интервал (min)", font=("Helvetica", 10)).grid(row=9, column=0)
    ttk.Label(param_frame, text="X интервал (max)", font=("Helvetica", 10)).grid(row=10, column=0)
    ttk.Label(param_frame, text="Y интервал (min)", font=("Helvetica", 10)).grid(row=11, column=0)
    ttk.Label(param_frame, text="Y интервал (max)", font=("Helvetica", 10)).grid(row=12, column=0)
    ttk.Label(param_frame, text="Ось X интервал", font=("Helvetica", 10)).grid(row=13, column=0)
    ttk.Label(param_frame, text="Ось Y интервал", font=("Helvetica", 10)).grid(row=14, column=0)

    x_interval_min = tk.DoubleVar(value=-3)
    x_interval_max = tk.DoubleVar(value=3)
    y_interval_min = tk.DoubleVar(value=-3)
    y_interval_max = tk.DoubleVar(value=3)
    x_axis_interval = tk.IntVar(value=1)
    y_axis_interval = tk.IntVar(value=1)

    ttk.Entry(param_frame, textvariable=x_interval_min).grid(row=9, column=1)
    ttk.Entry(param_frame, textvariable=x_interval_max).grid(row=10, column=1)
    ttk.Entry(param_frame, textvariable=y_interval_min).grid(row=11, column=1)
    ttk.Entry(param_frame, textvariable=y_interval_max).grid(row=12, column=1)
    ttk.Entry(param_frame, textvariable=x_axis_interval).grid(row=13, column=1)
    ttk.Entry(param_frame, textvariable=y_axis_interval).grid(row=14, column=1)

    ttk.Separator(param_frame, orient="horizontal").grid(row=15, column=0, columnspan=2, sticky="ew", pady=10)

    # Кнопка и результаты
    ttk.Label(param_frame, text="Выполнение и результаты", font=("Helvetica", 12)).grid(row=16, column=0, pady=10)
    ttk.Button(param_frame, text="Выполнить", command=run_optimization, style="My.TButton").grid(row=17, column=1, padx=10, pady=10)
    results_text = scrolledtext.ScrolledText(param_frame, wrap=tk.WORD, height=18, width=40, padx=2, state=tk.DISABLED)
    results_text.grid(row=17, column=0, padx=10)
