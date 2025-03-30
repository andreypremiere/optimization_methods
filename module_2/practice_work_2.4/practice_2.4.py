# Вариант 7
# Метод Гаусса-Зейделя
# 6x1^2 + x1x2 +3x2^2 -> min

def objective_function_1(x1, x2):
    return 2 * x2 ** 2 - 2 * x2 + x1 * x2 + 4 * x1 ** 2


def objective_function_2(x1, x2):
    return 4 * (x1 - 5) ** 2 + (x2 - 6) ** 2


def objective_function_3(x1, x2):
    return 2 * x1 ** 2 + x1 * x2 + x2 ** 2

def grad_f1_x1(x1, x2):
    return x2 + 8 * x1

def grad_f1_x2(x1, x2):
    return 4 * x2 - 2 + x1

def grad_f2_x1(x1, x2):
    return 8 * (x1 - 5)

def grad_f2_x2(x1, x2):
    return 2 * (x2 - 6)

def grad_f3_x1(x1, x2):
    return 4 * x1 + x2

def grad_f3_x2(x1, x2):
    return x1 + 2 * x2


def gauss_seidel_method(learning_rate=0.1, max_iter=1000, tol=1e-5, initial_guess=(10, 10), grad_f_x1=None, grad_f_x2=None, f=None):
    x1, x2 = initial_guess
    func_evals = 0

    for _ in range(max_iter):
        x1_new = x1 - learning_rate * grad_f_x1(x1, x2)
        x2_new = x2 - learning_rate * grad_f_x2(x1_new, x2)

        func_evals += 2

        if abs(x1_new - x1) < tol and abs(x2_new - x2) < tol:
            break

        x1, x2 = x1_new, x2_new

    return x1, x2, f(x1, x2), func_evals


x_inits = [(-20, -20), (10, 10), (100, 100)]
functions = [objective_function_1, objective_function_2, objective_function_3]
gradients = [grad_f1_x1, grad_f1_x2, grad_f2_x1, grad_f2_x2, grad_f3_x1, grad_f3_x2]

for i in range(len(functions)):
    for j in x_inits:
        x1, x2, f_min, evaluations = gauss_seidel_method(initial_guess=j, f=functions[i], grad_f_x1=gradients[i*2], grad_f_x2=gradients[i*2+1])
        print(f"alpha=0.1, x_init={j}, e=1e-5, Минимум: ({x1:.6f}, {x2:.6f}), Значение: {f_min:.6f}, Вычислений функции: {evaluations}")

