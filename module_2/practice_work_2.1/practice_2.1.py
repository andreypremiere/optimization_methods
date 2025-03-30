# Вариант 7
# Метод градиентного спуска с постоянным шагом
# 6x1^2 + x1x2 +3x2^2 -> min
import numpy as np


# Функции для исследования
def objective_function_1(x1, x2):
    return 2 * x2 ** 2 - 2 * x2 + x1 * x2 + 4 * x1 ** 2


def objective_function_2(x1, x2):
    return 4 * (x1 - 5) ** 2 + (x2 - 6) ** 2


def objective_function_3(x1, x2):
    return 2 * x1 ** 2 + x1 * x2 + x2 ** 2


def objective_function(x1, x2):
    return 6 * x1 ** 2 + x1 * x2 + 3 * x2 ** 2


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

def gradient(x1, x2):
    df_dx1 = 12 * x1 + x2
    df_dx2 = x1 + 6 * x2
    return np.array([df_dx1, df_dx2])

def gradient_descent(alpha=0.1, tol=1e-5, max_iters=1000, x_init=(50, 50), objective_function=None, gradient=None):
    x = np.array(x_init, dtype=float)
    history = [tuple(x)]
    func_evals = 0

    for _ in range(max_iters):
        grad = gradient(*x)
        func_evals += 1
        new_x = x - alpha * grad

        if np.linalg.norm(new_x - x) < tol:
            break

        x = new_x
        history.append(tuple(x))

    return x, objective_function(*x), history, func_evals


alphas = [0.01, 0.05, 0.1]
functions = [objective_function_1, objective_function_2, objective_function_3]
gradients = [gradient_1, gradient_2, gradient_3]


# for alpha in alphas:
#     x_min, f_min, history, evaluations = gradient_descent(alpha=alpha)
#     print(f"Alpha: {alpha}, Минимум: ({x_min[0]}, {x_min[1]}), Значение: {f_min}, Вычислений функции: {evaluations}")


for i in range(len(functions)):
    for j in alphas:
        x_min, f_min, history, evaluations = gradient_descent(alpha=j, objective_function=functions[i], gradient=gradients[i])
        print(f"alpha: {j}, x_init=(50, 50), e=1e-5, Минимум: ({x_min[0]:.6f}, {x_min[1]:.6f}), Значение: {f_min:.6f}, Вычислений функции: {evaluations}")