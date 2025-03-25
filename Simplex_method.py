import tkinter as tk
from tkinter import ttk
import numpy as np
import time
from tkinter import scrolledtext
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Simplex_method(frame, root, ax, canvas):
    def objective(x, coeffs):
        x1, x2 = x[0], x[1]
        return coeffs[0] * x1 ** 2 + coeffs[1] * x2 ** 2 + coeffs[2] * x1 * x2 + coeffs[3] * x1 + coeffs[4] * x2

    def constraints(coeffs_con):
        cons = []
        for i in range(0, len(coeffs_con), 3):
            a, b, c = coeffs_con[i], coeffs_con[i+1], coeffs_con[i+2]
            cons.append({'type': 'ineq', 
                        'fun': lambda x, a=a, b=b, c=c: c - (a * x[0] + b * x[1])})
        
        cons.append({'type': 'ineq', 'fun': lambda x: x[0]})
        cons.append({'type': 'ineq', 'fun': lambda x: x[1]})
        return cons

    def simplex_method(x0, coeffs_obj, coeffs_con, opt_type):
        points = []
        
        def callback(x_w):
            if opt_type == "minimize":
                value = objective(x_w, coeffs_obj)
            else:  # maximize
                value = -objective(x_w, coeffs_obj)
            points.append([x_w[0], x_w[1], value])
        
        if opt_type == "minimize":
            result = minimize(objective, x0, args=coeffs_obj,
                            constraints=constraints(coeffs_con),
                            callback=callback,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            final_value = result.fun
        else:  # maximize
            result = minimize(lambda x: -objective(x, coeffs_obj), x0,
                            constraints=constraints(coeffs_con),
                            callback=callback,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            final_value = -result.fun
        
        points.append([result.x[0], result.x[1], final_value])
        
        for iteration, point in enumerate(points):
            yield iteration, point

    def run_optimization():
        x0 = [x_var.get(), y_var.get()]
        coeffs_obj = [coeff_x2.get(), coeff_y2.get(), coeff_xy.get(), coeff_x.get(), coeff_y.get()]
        coeffs_con1 = [con1_a.get(), con1_b.get(), con1_c.get()]
        coeffs_con2 = [con2_a.get(), con2_b.get(), con2_c.get()]
        coeffs_con = coeffs_con1 + coeffs_con2
        delay = delay_var.get()
        opt_type = opt_type_var.get()

        x_cs, y_cs, z_cs = [], [], []
        ax.cla()

        x_range = np.linspace(x_interval_min.get(), x_interval_max.get(), 100)
        y_range = np.linspace(y_interval_min.get(), y_interval_max.get(), 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = objective([X, Y], coeffs_obj) if opt_type == "minimize" else -objective([X, Y], coeffs_obj)

        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Z')
        ax.set_xticks(np.arange(x_interval_min.get(), x_interval_max.get() + 1, x_axis_interval.get()))
        ax.set_yticks(np.arange(y_interval_min.get(), y_interval_max.get() + 1, y_axis_interval.get()))
        ax.set_title(f"Алгоритм Симплекс-метод ({'Минимизация' if opt_type == 'minimize' else 'Максимизация'})")

        results = []
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)

        for i, point in simplex_method(x0, coeffs_obj, coeffs_con, opt_type):
            x_cs.append(point[0])
            y_cs.append(point[1])
            z_cs.append(point[2])

            results.append((point[0], point[1], i, point[2]))
            ax.scatter(point[0], point[1], point[2], color='red', s=10)
            results_text.insert(tk.END,
                              f"Шаг {i}: Координаты ({point[0]:.2f}, {point[1]:.2f}), Значение: {point[2]:.7f}\n")
            results_text.yview_moveto(1)
            canvas.draw()
            root.update()
            time.sleep(delay)

        length = len(results) - 1
        ax.scatter(results[length][0], results[length][1], results[length][3], color='black', marker='x', s=60)
        results_text.insert(tk.END,
                          f"Результат ({'минимум' if opt_type == 'minimize' else 'максимум'}):\n"
                          f"Координаты ({results[length][0]:.8f}, {results[length][1]:.8f})\n"
                          f"Значение: {results[length][3]:.8f}\n")
        results_text.yview_moveto(1)
        results_text.config(state=tk.DISABLED)

    param_frame2 = frame

    # Начальные значения
    ttk.Label(param_frame2, text="Начальные значения", font=("Helvetica", 12)).grid(row=0, column=0, pady=15)
    ttk.Label(param_frame2, text="x₁ начальное").grid(row=1, column=0)
    ttk.Label(param_frame2, text="x₂ начальное").grid(row=2, column=0)
    ttk.Label(param_frame2, text="Задержка").grid(row=3, column=0)

    x_var = tk.DoubleVar(value=20)
    y_var = tk.DoubleVar(value=20)
    delay_var = tk.DoubleVar(value=0.5)

    ttk.Entry(param_frame2, textvariable=x_var).grid(row=1, column=1)
    ttk.Entry(param_frame2, textvariable=y_var).grid(row=2, column=1)
    ttk.Entry(param_frame2, textvariable=delay_var).grid(row=3, column=1)

    # ttk.Separator(param_frame2, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

    # Тип оптимизации
    # ttk.Label(param_frame2, text="Тип оптимизации", font=("Helvetica", 12)).grid(row=5, column=0, pady=10)
    opt_type_var = tk.StringVar(value="minimize")
    # ttk.Radiobutton(param_frame2, text="Минимизация", variable=opt_type_var, value="minimize").grid(row=6, column=0)
    # ttk.Radiobutton(param_frame2, text="Максимизация", variable=opt_type_var, value="maximize").grid(row=6, column=1)

    ttk.Separator(param_frame2, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

    # Коэффициенты целевой функции
    ttk.Label(param_frame2, text="Целевая функция", font=("Helvetica", 12)).grid(row=8, column=0, pady=10)
    ttk.Label(param_frame2, text="x₁²", font=("Helvetica", 14)).grid(row=9, column=0)
    ttk.Label(param_frame2, text="x₂²", font=("Helvetica", 14)).grid(row=10, column=0)
    ttk.Label(param_frame2, text="x₁x₂", font=("Helvetica", 14)).grid(row=11, column=0)
    ttk.Label(param_frame2, text="x₁", font=("Helvetica", 14)).grid(row=12, column=0)
    ttk.Label(param_frame2, text="x₂", font=("Helvetica", 14)).grid(row=13, column=0)

    coeff_x2 = tk.DoubleVar(value=2)
    coeff_y2 = tk.DoubleVar(value=3)
    coeff_xy = tk.DoubleVar(value=4)
    coeff_x = tk.DoubleVar(value=-6)
    coeff_y = tk.DoubleVar(value=-3)

    ttk.Entry(param_frame2, textvariable=coeff_x2).grid(row=9, column=1)
    ttk.Entry(param_frame2, textvariable=coeff_y2).grid(row=10, column=1)
    ttk.Entry(param_frame2, textvariable=coeff_xy).grid(row=11, column=1)
    ttk.Entry(param_frame2, textvariable=coeff_x).grid(row=12, column=1)
    ttk.Entry(param_frame2, textvariable=coeff_y).grid(row=13, column=1)

    ttk.Separator(param_frame2, orient="horizontal").grid(row=14, column=0, columnspan=2, sticky="ew", pady=10)

    # Ограничения
    ttk.Label(param_frame2, text="Ограничения", font=("Helvetica", 12)).grid(row=15, column=0, pady=10)
    
    ttk.Label(param_frame2, text="Первое: ax + by ≤ c").grid(row=16, column=0)
    con1_a = tk.DoubleVar(value=0)
    con1_b = tk.DoubleVar(value=0)
    con1_c = tk.DoubleVar(value=0)
    ttk.Entry(param_frame2, textvariable=con1_a).grid(row=16, column=1)
    ttk.Entry(param_frame2, textvariable=con1_b).grid(row=16, column=2)
    ttk.Entry(param_frame2, textvariable=con1_c).grid(row=16, column=3)

    ttk.Label(param_frame2, text="Второе: ax + by ≤ c").grid(row=17, column=0)
    con2_a = tk.DoubleVar(value=0)
    con2_b = tk.DoubleVar(value=0)
    con2_c = tk.DoubleVar(value=0)
    ttk.Entry(param_frame2, textvariable=con2_a).grid(row=17, column=1)
    ttk.Entry(param_frame2, textvariable=con2_b).grid(row=17, column=2)
    ttk.Entry(param_frame2, textvariable=con2_c).grid(row=17, column=3)

    ttk.Separator(param_frame2, orient="horizontal").grid(row=18, column=0, columnspan=4, sticky="ew", pady=10)

    # Параметры графика
    ttk.Label(param_frame2, text="Параметры графика", font=("Helvetica", 12)).grid(row=19, column=0, pady=10)
    ttk.Label(param_frame2, text="x₁ интервал (min)").grid(row=20, column=0)
    ttk.Label(param_frame2, text="x₁ интервал (max)").grid(row=21, column=0)
    ttk.Label(param_frame2, text="x₂ интервал (min)").grid(row=22, column=0)
    ttk.Label(param_frame2, text="x₂интервал (max)").grid(row=23, column=0)
    ttk.Label(param_frame2, text="Ось x₁ интервал").grid(row=24, column=0)
    ttk.Label(param_frame2, text="Ось x₂ интервал").grid(row=25, column=0)

    x_interval_min = tk.DoubleVar(value=-5)
    x_interval_max = tk.DoubleVar(value=5)
    y_interval_min = tk.DoubleVar(value=-5)
    y_interval_max = tk.DoubleVar(value=5)
    x_axis_interval = tk.IntVar(value=2)
    y_axis_interval = tk.IntVar(value=2)

    ttk.Entry(param_frame2, textvariable=x_interval_min).grid(row=20, column=1)
    ttk.Entry(param_frame2, textvariable=x_interval_max).grid(row=21, column=1)
    ttk.Entry(param_frame2, textvariable=y_interval_min).grid(row=22, column=1)
    ttk.Entry(param_frame2, textvariable=y_interval_max).grid(row=23, column=1)
    ttk.Entry(param_frame2, textvariable=x_axis_interval).grid(row=24, column=1)
    ttk.Entry(param_frame2, textvariable=y_axis_interval).grid(row=25, column=1)

    # Кнопка и результаты
    ttk.Button(param_frame2, text="Выполнить", command=run_optimization).grid(row=26, column=1, pady=10)
    ttk.Label(param_frame2, text="Результаты", font=("Helvetica", 12)).grid(row=26, column=0, pady=10)
    results_text = scrolledtext.ScrolledText(param_frame2, wrap=tk.WORD, height=10, width=40, state=tk.DISABLED)
    results_text.grid(row=27, column=0, columnspan=2, padx=10)