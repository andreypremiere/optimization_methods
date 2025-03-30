# Вариант 7
# Метод покоординатного градиентного спуска
# 6x1^2 + x1x2 +3x2^2 -> min

import numpy as np


def objective_function_1(x1, x2):
    return 2 * x2 ** 2 - 2 * x2 + x1 * x2 + 4 * x1 ** 2


def objective_function_2(x1, x2):
    return 4 * (x1 - 5) ** 2 + (x2 - 6) ** 2


def objective_function_3(x1, x2):
    return 2 * x1 ** 2 + x1 * x2 + x2 ** 2


def gradient_1(x1, x2):
    df_dx1 = x2 + 8 * x1
    df_dx2 = 4 * x2 - 2 + x1
    return np.array([df_dx1, df_dx2])


def gradient_2(x1, x2):
    df_dx1 = 8 * (x1 - 5)
    df_dx2 = 2 * (x2 - 6)
    return np.array([df_dx1, df_dx2])


def gradient_3(x1, x2):
    df_dx1 = 4 * x1 + x2
    df_dx2 = x1 + 2 * x2
    return np.array([df_dx1, df_dx2])


import numpy as np


def coordinate_search(x, index, max_iters=1000, tol=1e-5, objective_function=None, gradient=None):
    alpha = 1.0
    beta = 0.5
    direction = np.zeros_like(x)

    grad = gradient(*x)  # Исправлено: передаём x1, x2
    direction[index] = -1 if grad[index] > 0 else 1  # Исправлено: не использовать sign(0)

    while max_iters > 0:
        new_x = x + alpha * direction
        if objective_function(*new_x) < objective_function(*x) - tol * alpha * np.dot(direction, direction):
            return alpha
        alpha *= beta
        max_iters -= 1

    return alpha


def coordinate_descent(tol=1e-5, max_iters=1000, x_init=(10, 10), gradient=None, objective_function=None):
    x = np.array(x_init, dtype=float)
    history = [tuple(x)]
    func_evals = 0

    for _ in range(max_iters):
        for i in range(len(x)):
            grad = gradient(*x)  # Исправлено: передаём x1, x2
            if np.abs(grad[i]) < tol:
                continue

            alpha = coordinate_search(x, i, max_iters, tol, objective_function, gradient)  # Исправлено

            direction = np.zeros_like(x)
            direction[i] = -1 if grad[i] > 0 else 1  # Исправлено: избегаем sign(0)

            new_x = x + alpha * direction
            func_evals += 1

            if np.linalg.norm(new_x - x) < tol:
                return x, objective_function(*x), history, func_evals

            x = new_x
            history.append(tuple(x))

    return x, objective_function(*x), history, func_evals


x_inits = [(-20, -20), (10, 10), (100, 100)]
functions = [objective_function_1, objective_function_2, objective_function_3]
gradients = [gradient_1, gradient_2, gradient_3]

for i in range(len(functions)):
    for j in x_inits:
        x_min, f_min, history, evaluations = coordinate_descent(x_init=j, objective_function=functions[i], gradient=gradients[i])
        print(f"x_init={j}, e=1e-5, Минимум: ({x_min[0]:.6f}, {x_min[1]:.6f}), Значение: {f_min:.6f}, Вычислений функции: {evaluations}")
