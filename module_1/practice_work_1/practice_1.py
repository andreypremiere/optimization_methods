import time


def fibonacci_method(func, a, b, epsilon):
    global f_evals

    # Определение необходимого количества чисел Фибоначчи
    fib = [0, 1]
    while fib[-1] < (b - a) / epsilon:
        fib.append(fib[-1] + fib[-2])

    n = len(fib) - 1  # Количество шагов

    # Инициализация точек
    x1 = a + (fib[n - 2] / fib[n]) * (b - a)
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)

    f1, f2 = func(x1), func(x2)
    f_evals += 2  # Два вызова функции

    for _ in range(n - 2):
        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = a + (fib[n - 1] / fib[n]) * (b - a)
            f2 = func(x2)
            f_evals += 1  # Новый вызов функции
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = a + (fib[n - 2] / fib[n]) * (b - a)
            f1 = func(x1)
            f_evals += 1  # Новый вызов функции
        n -= 1

    return (a + b) / 2, func((a + b) / 2)

# Основная программа
if __name__ == "__main__":
    # Сбрасываем счетчик вызовов
    f_evals = 0

    # Целевая функция
    def target_function(x):
        global f_evals
        f_evals += 1
        return 2 * (x ** 2) - 2 * x + 14

    # Пример использования
    a, b = -6, 6
    epsilon = 1e-10
    start_time = time.time()
    x_min, f_min = fibonacci_method(target_function, a, b, epsilon)
    end_time = time.time()

    print(f"Минимум функции находится в точке: x = {x_min}")
    print(f"Значение функции: {f_min}")
    print(f"Function evaluations: {f_evals}")
    print(f'Time execution: {end_time - start_time:.20f}')
