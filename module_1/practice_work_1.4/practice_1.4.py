# Вариант 7
# Метод Нелдер-Мида
import time

import numpy as np


def f(x):
    """Целевая функция."""
    return (x + 1) ** 2 + 2 * x + 1


def nelder_mead(f, x0, alpha=1, gamma=2, rho=0.5, sigma=0.5, tol=1e-10, max_iter=10000):
    """
    Метод оптимизации Нелдера-Мида.

    Parameters:
    - f: Функция для минимизации.
    - x0: Начальное приближение (список из 2 точек).
    - alpha: Коэффициент отражения.
    - gamma: Коэффициент расширения.
    - rho: Коэффициент сжатия.
    - sigma: Коэффициент уменьшения симплекса.
    - tol: Допустимая погрешность.
    - max_iter: Максимальное число итераций.

    Returns:
    - Оптимальное значение x и значение функции.
    """
    x1, x2 = x0
    simplex = np.array([[x1], [x2]])
    f_values = np.array([f(x1), f(x2)])

    for _ in range(max_iter):
        indices = np.argsort(f_values)
        simplex, f_values = simplex[indices], f_values[indices]

        centroid = np.mean(simplex[:-1], axis=0)

        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)

        if f_values[0] <= fr < f_values[-2]:
            simplex[-1], f_values[-1] = xr.item(), fr.item()
        elif fr < f_values[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], f_values[-1] = xe.item(), fe.item()
            else:
                simplex[-1], f_values[-1] = xr.item(), fr.item()
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < f_values[-1]:
                simplex[-1], f_values[-1] = xc.item(), fc.item()
            else:
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                f_values[1:] = [f(x) for x in simplex[1:]]

        if np.max(np.abs(simplex - centroid)) < tol:
            break

    return simplex[0][0], f_values[0]


start_timestamp = time.time()
x_min, f_min = nelder_mead(f, x0=[-6, 6])
end_timestamp = time.time()

print(f"Optimal x: {x_min}")
print(f'Time execution: {end_timestamp - start_timestamp:.20f}')
