import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from ipywidgets import interact, FloatLogSlider


np.random.seed(42)
X = np.sort(np.random.rand(20))
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, len(X))
X_plot = np.linspace(0, 1, 100)


def plot_ridge(alpha):
    # Создаем полиномиальную регрессию высокой степени (склонную к переобучению)
    model = make_pipeline(PolynomialFeatures(degree=12), Ridge(alpha=alpha))
    model.fit(X[:, np.newaxis], y)
    y_plot = model.predict(X_plot[:, np.newaxis])

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Данные (с шумом)')
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), color='green', linestyle='--', label='Истинная функция')
    plt.plot(X_plot, y_plot, color='blue', linewidth=2, label=f'L2 Регуляризация (alpha={alpha:.2e})')

    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.title("Как L2-штраф усмиряет модель")
    plt.grid(True, alpha=0.3)
    plt.show()


from sklearn.linear_model import Lasso

def plot_lasso(alpha):
    # Создаем полиномиальную регрессию с L1 штрафом
    # Увеличим max_iter, так как Lasso может долго сходиться на высоких степенях
    model = make_pipeline(PolynomialFeatures(degree=12), Lasso(alpha=alpha, max_iter=100000))
    model.fit(X[:, np.newaxis], y)
    y_plot = model.predict(X_plot[:, np.newaxis])

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Данные (с шумом)')
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), color='green', linestyle='--', label='Истинная функция')
    plt.plot(X_plot, y_plot, color='purple', linewidth=2, label=f'L1 Регуляризация (alpha={alpha:.2e})')

    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.title("Lasso (L1): Зануление лишних коэффициентов")
    plt.grid(True, alpha=0.3)
    plt.show()



from sklearn.linear_model import ElasticNet

def plot_elastic_net(alpha, l1_ratio=0.5):
    # l1_ratio=1 это Lasso, l1_ratio=0 это Ridge
    model = make_pipeline(PolynomialFeatures(degree=12),
                          ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000))
    model.fit(X[:, np.newaxis], y)
    y_plot = model.predict(X_plot[:, np.newaxis])

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Данные (с шумом)')
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), color='green', linestyle='--', label='Истинная функция')
    plt.plot(X_plot, y_plot, color='orange', linewidth=2,
             label=f'ElasticNet (alpha={alpha:.2e}, l1_ratio={l1_ratio})')

    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.title("ElasticNet: Баланс между отбором и сглаживанием")
    plt.grid(True, alpha=0.3)
    plt.show()

def runge_example():
    """
    Демонстрация переобучения на примере аппроксимации функции Рунге.
    Анимация показывает, как с увеличением степени полинома происходит переобучение.
    """
    # Функция Рунге
    def runge_function(x):
        return 1 / (1 + 25 * x**2)

    # Генерация данных
    np.random.seed(42)
    n_samples = 20
    X = np.linspace(-1, 1, n_samples)
    y_true = runge_function(X)
    y = y_true + np.random.normal(0, 0.05, n_samples)  # Добавляем шум

    # Используем только обучающую выборку
    X_train = X
    y_train = y

    # Для плавной визуализации
    x_plot = np.linspace(-1, 1, 500)
    y_plot_true = runge_function(x_plot)

    # Списки для хранения истории
    degrees = []
    train_errors = []
    overfitting_detected = False
    best_degree = 1
    min_train_error = float('inf')

    # Основной цикл анимации
    for degree in range(1, 41):
        # Очищаем вывод для анимации
        clear_output(wait=True)

        # Создаем новую фигуру для каждого кадра
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Переобучение при аппроксимации функции Рунге (Степень: {degree})',
                     fontsize=16, fontweight='bold')

        # Настройка первого графика
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Настройка второго графика
        ax2.set_xlabel('Степень полинома', fontsize=12)
        ax2.set_ylabel('Ошибка (MSE)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 41)
        ax2.set_ylim(0, 0.1)

        # Исходные данные (только обучающая выборка)
        ax1.scatter(X_train, y_train, color='blue', s=60, label='Обучающая выборка',
                    zorder=5, alpha=0.7)
        ax1.plot(x_plot, y_plot_true, 'k--', label='Истинная функция',
                 alpha=0.7, linewidth=2.5)

        # Создаем и обучаем модель
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train.reshape(-1, 1), y_train)

        # Предсказания
        y_pred_plot = model.predict(x_plot.reshape(-1, 1))
        y_pred_train = model.predict(X_train.reshape(-1, 1))

        # Вычисление ошибок (только на обучающей выборке)
        train_error = np.mean((y_train - y_pred_train) ** 2)

        # Сохраняем ошибки
        degrees.append(degree)
        train_errors.append(train_error)

        # Обновляем лучшую степень (минимальная ошибка на обучении)
        if train_error < min_train_error:
            min_train_error = train_error
            best_degree = degree

        # Рисуем аппроксимацию
        ax1.plot(x_plot, y_pred_plot, 'b-', label='Аппроксимация',
                 linewidth=2, alpha=0.9)

        # Рисуем кривую обучения (только ошибка на обучении)
        ax2.plot(degrees, train_errors, 'b-', label='Ошибка на обучении',
                 linewidth=2.5, marker='o', markersize=5)

        # Проверка на переобучение
        title_color = 'black'
        warning_text = ''

        if len(train_errors) >= 3:
            # Проверяем, когда ошибка на обучении становится очень маленькой
            # и полином начинает сильно колебаться
            if train_error < 0.001 and degree > 20:
                if not overfitting_detected:
                    overfitting_detected = True
                    ax1.set_facecolor((1, 0.9, 0.9))
                    title_color = 'red'
                    warning_text = 'СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ!'

                    # Добавляем вертикальную линию на втором графике
                    ax2.axvline(x=degree, color='red', linestyle=':', alpha=0.7,
                                linewidth=2, label='Начало переобучения')

            # Альтернативный критерий: когда полином имеет слишком много экстремумов
            # Считаем количество пересечений с истинной функцией
            y_pred_smooth = model.predict(x_plot.reshape(-1, 1))
            diff_sign = np.diff(np.sign(y_pred_smooth - y_plot_true))
            num_crossings = np.sum(diff_sign != 0)

            if num_crossings > degree and degree > 15:
                if not overfitting_detected:
                    overfitting_detected = True
                    ax1.set_facecolor((1, 0.95, 0.9))
                    title_color = 'orange'
                    warning_text = 'ПЕРЕОБУЧЕНИЕ (много колебаний)'

                    # Добавляем вертикальную линию на втором графике
                    ax2.axvline(x=degree, color='orange', linestyle=':', alpha=0.7,
                                linewidth=2, label='Начало переобучения')

        # Обновляем заголовок с цветом
        ax1.set_title(f'Аппроксимация полиномом {degree}-й степени',
                      fontsize=14, fontweight='bold', color=title_color)
        ax2.set_title('Кривая обучения', fontsize=14, fontweight='bold')

        # Легенды
        ax1.legend(loc='upper right', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        # Показываем график
        display(fig)

        # Вывод информации в консоль
        print(f"Шаг {degree}/40: Степень {degree}, Ошибка = {train_error:.6f}")
        if warning_text:
            print(f"   {warning_text}")

        # Пауза для анимации (1 секунда)
        time.sleep(1.0)

        # Закрываем фигуру, чтобы не накапливать в памяти
        plt.close(fig)

    # Первый график - лучшая аппроксимация
    model_best = make_pipeline(
        PolynomialFeatures(best_degree), LinearRegression())
    model_best.fit(X_train.reshape(-1, 1), y_train)
    y_pred_best = model_best.predict(x_plot.reshape(-1, 1))

    plt.tight_layout()
    plt.show()
