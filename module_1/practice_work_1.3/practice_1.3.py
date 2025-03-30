# Вариант 7
# Метод Хука-Дживса

import numpy as np
import time

def f(x):
    """Целевая функция."""
    return (x + 1) ** 2 + 2 * x + 1


def hooke_jeeves(f, x0, step_size=1.0, alpha=2.0, tol=1e-10, max_iter=10000, bounds=(-6, 6)):
    """
    Метод оптимизации Хука-Дживса.

    Parameters:
    - f: Функция, которую нужно минимизировать.
    - x0: Начальная точка.
    - bounds: Границы: (нижняя, верхняя).
    - step_size: Шаг итерации.
    - alpha: Коэффициент ускорения.
    - tol: Допустимая погрешность для остановки.
    - max_iter: Максимальное количество итераций.

    Returns:
    - Оптимальное значение x и значение функции.
    """
    x = np.clip(x0, *bounds)
    best_x = x
    best_f = f(x)

    for _ in range(max_iter):
        new_x = best_x
        for direction in [-1, 1]:
            candidate_x = np.clip(best_x + direction * step_size, *bounds)
            candidate_f = f(candidate_x)

            if candidate_f < best_f:
                best_x, best_f = candidate_x, candidate_f

        if not np.all(new_x == best_x):
            pattern_x = np.clip(best_x + alpha * (best_x - new_x), *bounds)
            pattern_f = f(pattern_x)

            if pattern_f < best_f:
                best_x, best_f = pattern_x, pattern_f
            else:
                step_size /= 2
        else:
            step_size /= 2

        if step_size < tol:
            break

    return best_x, best_f


start_timestamp = time.time()
x_min, f_min = hooke_jeeves(f, x0=0)
end_timestamp = time.time()

print(f"Optimal x: {x_min}")
print(f'Time execution: {end_timestamp - start_timestamp:.20f}')
