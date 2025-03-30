# Вариант 7
# Метод Нелдера-Мида

import numpy as np
import time

f_evals = 0

# 1. 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2
# 2. 2 * x[0]**2 + x[0] * x[1] + x[1]**2
# 3. 2*x[1]**2 - 2*x[1] + x[0]*x[1] + 4*x[0]**2

def f(x):
    global f_evals
    f_evals += 1
    return 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2


def nelder_mead(f, x_start, alpha=1, gamma=2, rho=0.5, sigma=0.5, tol=1e-1, max_iter=10000):
    # Инициализация симплекса (3 точки в 2D-пространстве)
    n = len(x_start)
    simplex = np.array([x_start, x_start + np.eye(n)[0], x_start + np.eye(n)[1]])
    f_values = np.array([f(x) for x in simplex])

    for _ in range(max_iter):
        # Сортировка точек симплекса по значению функции
        indices = np.argsort(f_values)
        simplex = simplex[indices]
        f_values = f_values[indices]

        # Проверка условия остановки
        if np.std(f_values) < tol:
            break

        # Вычисление центра масс без худшей точки
        x_bar = np.mean(simplex[:-1], axis=0)

        # Отражение
        x_r = x_bar + alpha * (x_bar - simplex[-1])
        f_r = f(x_r)

        if f_values[0] <= f_r < f_values[-2]:  # Улучшение, но не лучшее
            simplex[-1] = x_r
            f_values[-1] = f_r
        elif f_r < f_values[0]:  # Лучшее значение, пробуем растяжение
            x_e = x_bar + gamma * (x_r - x_bar)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        else:  # Сжатие
            x_c = x_bar + rho * (simplex[-1] - x_bar)
            f_c = f(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:  # Редукция
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                f_values[1:] = [f(x) for x in simplex[1:]]

    return simplex[0], f_values[0]


if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])

    start_time = time.time()
    x_min, f_min = nelder_mead(f, x0)
    end_time = time.time()

    print(f"Оптимальное x: {x_min}")
    print(f"Минимальное значение функции: {f_min}")
    print(f"Количество вызовов функции: {f_evals}")
    print(f"Время выполнения: {end_time - start_time:.20f}")
