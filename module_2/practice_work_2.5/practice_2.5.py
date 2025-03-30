import numpy as np
from scipy.optimize import line_search

def dfp_method(f, grad_f, x0, tol=1e-5, max_iter=1000):
    """Метод Дэвидона-Флетчера-Пауэлла (DFP) для поиска минимума"""
    global function_calls
    function_calls = 0  # Счетчик вызовов функции

    n = len(x0)
    x_k = np.array(x0, dtype=float)
    H_k = np.eye(n)  # Инициализация приближенной матрицы Гессе

    for _ in range(max_iter):
        grad_k = grad_f(x_k)
        if np.linalg.norm(grad_k) < tol:
            break  # Критерий остановки

        p_k = -H_k @ grad_k  # Направление поиска

        # Линейный поиск для выбора оптимального шага
        alpha_k = line_search(f, grad_f, x_k, p_k)[0]
        if alpha_k is None:
            alpha_k = 1e-4  # Если поиск не удался, используем небольшой шаг

        x_next = x_k + alpha_k * p_k  # Обновляем точку
        s_k = x_next - x_k  # Смещение
        y_k = grad_f(x_next) - grad_k  # Изменение градиента

        # Обновление матрицы Гессе по DFP
        rho_k = 1.0 / np.dot(y_k, s_k) if np.dot(y_k, s_k) != 0 else 1e-8
        H_k = H_k + rho_k * np.outer(s_k, s_k) - (H_k @ np.outer(y_k, s_k) + np.outer(s_k, y_k) @ H_k) * rho_k

        x_k = x_next  # Переход к новой итерации

    return x_k, f(x_k), function_calls

# Определяем функции и их градиенты

# 1) f(x1, x2) = 2 * x2^2 - 2 * x2 + x1 * x2 + 4 * x1^2
def f1(x):
    global function_calls
    function_calls += 1
    x1, x2 = x
    return 2 * x2**2 - 2 * x2 + x1 * x2 + 4 * x1**2

def grad_f1(x):
    x1, x2 = x
    df_dx1 = x2 + 8 * x1
    df_dx2 = 4 * x2 - 2 + x1
    return np.array([df_dx1, df_dx2])

# 2) f(x1, x2) = 4 * (x1 - 5)^2 + (x2 - 6)^2
def f2(x):
    global function_calls
    function_calls += 1
    x1, x2 = x
    return 4 * (x1 - 5)**2 + (x2 - 6)**2

def grad_f2(x):
    x1, x2 = x
    df_dx1 = 8 * (x1 - 5)
    df_dx2 = 2 * (x2 - 6)
    return np.array([df_dx1, df_dx2])

# 3) f(x1, x2) = 2 * x1^2 + x1 * x2 + x2^2
def f3(x):
    global function_calls
    function_calls += 1
    x1, x2 = x
    return 2 * x1**2 + x1 * x2 + x2**2

def grad_f3(x):
    x1, x2 = x
    df_dx1 = 4 * x1 + x2
    df_dx2 = x1 + 2 * x2
    return np.array([df_dx1, df_dx2])

# Начальная точка
x0 = np.array([-20, -20])

# Запуск метода для всех функций
for i, (f, grad_f) in enumerate([(f1, grad_f1), (f2, grad_f2), (f3, grad_f3)], start=1):
    result_x, result_f, calls = dfp_method(f, grad_f, x0)
    print(f"\nФункция {i}:")
    print(f"Найденный минимум в точке: ({result_x[0]:.6f}, {result_x[1]:.6f})")
    print(f"Значение функции в минимуме: {result_f:.6f}")
    print(f"Количество вызовов целевой функции: {calls}")
