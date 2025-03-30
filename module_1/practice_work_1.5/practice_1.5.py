import numpy as np
import time

# 1. 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2
# 2. 2 * x[0]**2 + x[0] * x[1] + x[1]**2
# 3. 2*x[1]**2 - 2*x[1] + x[0]*x[1] + 4*x[0]**2

f_evals = 0


def f(x):
    """Функция, которую мы минимизируем."""
    global f_evals
    f_evals += 1
    return 2*x[1]**2 - 2*x[1] + x[0]*x[1] + 4*x[0]**2


def rosenbrock_method(f, x_start, alpha=0.01, beta=0.5, epsilon=1e-10, max_iter=10000):
    """
    Реализация метода Розенброка для минимизации функции.

    Параметры:
    - f: Целевая функция
    - x_start: Начальная точка
    - alpha: Начальный размер шага
    - beta: Коэффициент уменьшения шага
    - epsilon: Точность остановки
    - max_iter: Максимальное количество итераций

    Возвращает:
    - Оптимальную точку x
    - Значение функции в этой точке
    """
    global f_evals
    f_evals = 0  # Сбрасываем счетчик вызовов

    x = np.copy(x_start)
    n = len(x)
    directions = np.eye(n)  # Ортогональные базисные направления
    step_sizes = np.full(n, alpha)  # Размеры шагов для каждого направления

    for _ in range(max_iter):
        x_prev = np.copy(x)

        # Движение по каждому направлению
        for i in range(n):
            x_test = x + step_sizes[i] * directions[i]
            if f(x_test) < f(x):
                x = x_test
            else:
                x_test = x - step_sizes[i] * directions[i]
                if f(x_test) < f(x):
                    x = x_test
                else:
                    step_sizes[i] *= beta  # Уменьшаем шаг, если не нашли улучшения

        # Проверка на сходимость
        if np.linalg.norm(x - x_prev) < epsilon:
            break

    return x, f(x)


# Запуск оптимизации
if __name__ == "__main__":
    x_start = np.array([0.0, 0.0])

    start_time = time.time()
    x_min, f_min = rosenbrock_method(f, x_start)
    end_time = time.time()

    print(f"Оптимальное x: {x_min}")
    print(f"Минимальное значение функции: {f_min}")
    print(f"Количество вызовов функции: {f_evals}")
    print(f"Время выполнения: {end_time - start_time:.20f}")
