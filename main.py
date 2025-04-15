import sys
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gradient_descent import GradientDescentAlgorithm
from Simplex_method import Simplex_method
from LR3.LR3 import GeneticAlgorithm
from LR4.particleswarm.LR4 import ParticleSwarmAlgorithm
# Функция для корректного выхода из программы
def on_closing():
    root.quit()   # Завершает главный цикл
    root.destroy()  # Закрывает окно
    sys.exit()  # Полностью завершает процесс

# Создание окна приложения
root = tk.Tk()
root.title("Методы поисковой оптимизации")
root.protocol("WM_DELETE_WINDOW", on_closing)  # Обработчик закрытия окна

notebook = ttk.Notebook(root)
notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Инициализация графика при запуске программы
fig = plt.figure(figsize=(8, 9))  # Установка размеров фигуры (ширина, высота)
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, padx=20)

# Вкладка для лр1
param_frame = ttk.Frame(notebook, padding=(15, 0))
notebook.add(param_frame, text="Градиентный спуск")
GradientDescentAlgorithm(param_frame, root, ax, canvas)

# Вкладка для лр2
param_frame2 = ttk.Frame(notebook)
notebook.add(param_frame2, text="Решение КП")
Simplex_method(param_frame2, root, ax, canvas)

# Вкладка для лр3
param_frame3 = ttk.Frame(notebook)
notebook.add(param_frame3, text="Генетический алгоритм")
GeneticAlgorithm(param_frame3,root,ax,canvas)

# Вкладка для лр4
param_frame4 = ttk.Frame(notebook)
notebook.add(param_frame4, text="ЛР4")
ParticleSwarmAlgorithm(param_frame4,root,ax,canvas)

root.mainloop()
