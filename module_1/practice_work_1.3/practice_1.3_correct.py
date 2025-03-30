import numpy as np
import time


# Определяем целевую функцию
def f(x):
    return 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2


# Метод Хука-Дживса
def hooke_jeeves(f, x0, step_size=1.0, alpha=0.5, tol=1e-10, max_iter=10000):
    x = np.array(x0, dtype=float)
    fx = f(x)
    count_eval = 1

    while step_size > tol and count_eval < max_iter:
        # Исследовательский шаг
        new_x = np.copy(x)
        for i in range(len(x)):
            trial_x = np.copy(new_x)
            trial_x[i] += step_size  # Пробуем шаг вперёд
            f_trial = f(trial_x)
            count_eval += 1

            if f_trial < fx:
                new_x = trial_x
                fx = f_trial
            else:
                trial_x[i] -= 2 * step_size  # Пробуем шаг назад
                f_trial = f(trial_x)
                count_eval += 1
                if f_trial < fx:
                    new_x = trial_x
                    fx = f_trial

        # Проверяем, изменились ли координаты
        if np.all(new_x == x):
            step_size *= alpha  # Уменьшаем шаг
        else:
            x = new_x

    return x, fx, count_eval


# Начальная точка
x0 = [-6, 6]

start_time = time.time()
x_min, f_min, eval_count = hooke_jeeves(f, x0)
end_time = time.time()

exec_time = end_time - start_time

print(f"Минимум найден в точке: {x_min}")
print(f"Значение функции в минимуме: {f_min}")
print(f"Количество вычислений функции: {eval_count}")
print(f"Время выполнения (секунды): {exec_time:.20f}")
