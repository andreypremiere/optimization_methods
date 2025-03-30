import numpy as np
import time

# Глобальная переменная для подсчёта вызовов f(x)
f_evals = 0

def f(x):
    """Целевая функция."""
    global f_evals
    f_evals += 1  # Увеличиваем счетчик вызовов
    return 2*x**2 - 2*x + 14

def df(x):
    """Производная функции"""
    return 4*x - 2

def pauwels_optimization(f, df, x0, bounds, alpha=0.1, tol=1e-10, max_iter=10000):
    """
    Метод оптимизации Pauwels.

    Parameters:
    - f: Функция, которую нужно минимизировать.
    - df: Производная функции.
    - x0: Начальное приближение.
    - bounds: Границы: (нижняя, верхняя).
    - alpha: Размер шага.
    - tol: Допустимая погрешность для остановки.
    - max_iter: Максимальное количество итераций.

    Returns:
    - Оптимальное значение x и значение функции.
    """
    x = x0
    lower, upper = bounds

    for i in range(max_iter):
        grad = df(x)
        x_new = x - alpha * grad
        x_new = np.clip(x_new, lower, upper)

        if abs(f(x_new) - f(x)) < tol:
            break

        x = x_new

    return x, f(x)

# Запуск алгоритма и измерение времени
start_timestamp = time.time()
x_min, f_min = pauwels_optimization(f, df, x0=0, bounds=(-6, 6))
end_timestamp = time.time()

# Вывод результатов
print(f"Optimal x: {x_min}")
print(f"Function value: {f_min}")
print(f"Function evaluations: {f_evals}")
print(f'Time execution: {end_timestamp - start_timestamp:.20f}')
