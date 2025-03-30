import numpy as np

func_calls = 0


def f(x):
    global func_calls
    func_calls += 1
    return (x[0] - 2) ** 2 + (x[1] - 5) ** 2


def gradient(x):
    df_dx1 = 2 * (x[0] - 2)
    df_dx2 = 2 * (x[1] - 5)
    return np.array([df_dx1, df_dx2])


def hessian():
    return np.array([[2, 0], [0, 2]])


def f1(x):
    global func_calls
    func_calls += 1
    return 2 * x[1] ** 2 - 2 * x[1] + x[0] * x[1] + 4 * x[0] ** 2


def gradient1(x):
    df_dx1 = x[1] + 8 * x[0]
    df_dx2 = 4 * x[1] - 2 + x[0]
    return np.array([df_dx1, df_dx2])


def hessian1():
    return np.array([[8, 1], [1, 4]])


def f2(x):
    global func_calls
    func_calls += 1
    return 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2


def gradient2(x):
    df_dx1 = 8 * (x[0] - 5)
    df_dx2 = 2 * (x[1] - 6)
    return np.array([df_dx1, df_dx2])


def hessian2():
    return np.array([[8, 0], [0, 2]])


def f3(x):
    global func_calls
    func_calls += 1
    return 2 * x[0] ** 2 + x[0] * x[1] + x[1] ** 2


def gradient3(x):
    df_dx1 = 4 * x[0] + x[1]
    df_dx2 = x[0] + 2 * x[1]
    return np.array([df_dx1, df_dx2])


def hessian3():
    return np.array([[4, 1], [1, 2]])


def newton_method(x_init, tol=1e-6, max_iter=1000, hessian=None, f=None, gradient=None):
    global func_calls
    func_calls = 0

    x = np.array(x_init, dtype=float)
    H_inv = np.linalg.inv(hessian())

    for i in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < tol:
            break
        x = x - H_inv @ grad

    return x, f(x)


target_functions = [f1, f2, f3]
gradients = [gradient1, gradient2, gradient3]
hessians = [hessian1, hessian2, hessian3]


x0 = [[100, 100], [10, 10], [-20, -20]]

for j in range(len(target_functions)):
    for i in x0:
        minimum, f_min = newton_method(i, f=target_functions[j], gradient=gradients[j], hessian=hessians[j])
        print(f"x_init={i}, tol=1e-6, Минимум: ({minimum[0]:.6f}, {minimum[1]:.6f}), \nЗначение функции: {f_min:.6f}, "
              f"Количество вызовов целевой ф-и: {func_calls}")
    print()
