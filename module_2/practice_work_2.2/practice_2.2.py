# Вариант 7
# Метод наискорейшего градиентного спуска
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


def objective_function(x1, x2):
    return 6 * x1 ** 2 + x1 * x2 + 3 * x2 ** 2


def gradient(x1, x2):
    df_dx1 = 12 * x1 + x2
    df_dx2 = x1 + 6 * x2
    return np.array([df_dx1, df_dx2])


def line_search(x, grad, max_iters=1000, tol=1e-10):
    alpha = 1.0
    beta = 0.5
    while max_iters > 0:
        new_x = x - alpha * grad
        if objective_function(*new_x) < objective_function(*x) - tol * alpha * np.dot(grad, grad):
            return alpha
        alpha *= beta
        max_iters -= 1
    return alpha


def steepest_gradient_descent(tol=1e-10, max_iters=1000, x_init=(50, 50), gradient=None, objective_function=None):
    x = np.array(x_init, dtype=float)
    history = [tuple(x)]
    func_evals = 0

    for _ in range(max_iters):
        grad = gradient(*x)
        func_evals += 1

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        alpha = line_search(x, grad, tol=tol)
        new_x = x - alpha * grad

        if np.linalg.norm(new_x - x) < tol:
            break

        x = new_x
        history.append(tuple(x))

    return x, objective_function(*x), history, func_evals


x_inits = [(-20, -20), (10, 10), (100, 100)]
functions = [objective_function_1, objective_function_2, objective_function_3]
gradients = [gradient_1, gradient_2, gradient_3]

# for i in x_inits:
#     x_min, f_min, history, evaluations = steepest_gradient_descent(x_init=i)
#     print(f"Наискорейший градиентный спуск: Начальная точка: {i}, Минимум: ({x_min[0]}, {x_min[1]}), "
#           f"\nЗначение: {f_min}, Вычислений функции: {evaluations}")


for i in range(len(functions)):
    for j in x_inits:
        x_min, f_min, history, evaluations = steepest_gradient_descent(x_init=j, objective_function=functions[i], gradient=gradients[i])
        print(f"x_init={j}, e=1e-5, Минимум: ({x_min[0]:.6f}, {x_min[1]:.6f}), Значение: {f_min:.6f}, Вычислений функции: {evaluations}")
