import numpy as np
from scipy.optimize import line_search

# Количество вызовов целевой функции
function_calls = 0

def f(x):
    """Целевая функция: пример f(x, y) = (x-2)^2 + (y-3)^2"""
    global function_calls
    function_calls += 1
    return (x[0] - 2)**2 + (x[1] - 3)**2

def grad_f(x):
    """Градиент функции"""
    df_dx = 2 * (x[0] - 2)
    df_dy = 2 * (x[1] - 3)
    return np.array([df_dx, df_dy])

def dfp_method(f, grad_f, x0, tol=1e-6, max_iter=100):
    """Реализация метода DFP для поиска минимума функции"""
    n = len(x0)
    x_k = np.array(x0, dtype=float)
    H_k = np.eye(n)  # Начальная матрица приближенного Гессиана

    for _ in range(max_iter):
        grad_k = grad_f(x_k)
        if np.linalg.norm(grad_k) < tol:
            break  # Критерий остановки

        # Определение направления поиска
        p_k = -H_k @ grad_k

        # Линейный поиск для определения оптимального шага α_k
        alpha_k = line_search(f, grad_f, x_k, p_k)[0]
        if alpha_k is None:
            alpha_k = 1e-4  # Если линейный поиск не дал результата, берем маленький шаг

        # Обновление точки
        x_next = x_k + alpha_k * p_k
        s_k = x_next - x_k  # Смещение
        y_k = grad_f(x_next) - grad_k  # Изменение градиента

        # Обновление приближенной матрицы Гессе по формуле DFP
        rho_k = 1.0 / np.dot(y_k, s_k) if np.dot(y_k, s_k) != 0 else 1e-8
        H_k = H_k + rho_k * np.outer(s_k, s_k) - (H_k @ np.outer(y_k, s_k) + np.outer(s_k, y_k) @ H_k) * rho_k

        x_k = x_next  # Переход к следующей итерации

    return x_k, f(x_k), function_calls

# Начальная точка
x0 = np.array([0.0, 0.0])

# Запуск метода
result_x, result_f, calls = dfp_method(f, grad_f, x0)

# Вывод результатов
print(f"Найденный минимум в точке: {result_x}")
print(f"Значение функции в минимуме: {result_f}")
print(f"Количество вызовов целевой функции: {calls}")
