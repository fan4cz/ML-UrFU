import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def run_rf_animation():

    # 1. Данные
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    f_x = np.sin(X).ravel()
    y = f_x + np.random.normal(0, 0.3, X.shape[0])
    X_test = np.linspace(0, 5, 500)[:, np.newaxis]
    f_test = np.sin(X_test).ravel()

    n_trees = 100
    all_preds = []
    ensemble_variances = []
    biases = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    def update(i):
        ax1.clear()
        ax2.clear()

        # Обучаем дерево (Bagging + Random Features)
        idx = np.random.choice(range(len(X)), size=len(X), replace=True)
        tree = DecisionTreeRegressor(max_depth=6, max_features=1)
        tree.fit(X[idx], y[idx])

        # .ravel() гарантирует форму (500,), а не (500, 1)
        current_tree_pred = tree.predict(X_test).ravel()
        all_preds.append(current_tree_pred)

        # Текущий ансамбль
        ensemble_pred = np.mean(all_preds, axis=0)

        # Метрики
        current_bias = np.mean((ensemble_pred - f_test)**2)

        # Расчет дисперсии именно АНСАМБЛЯ (уменьшается с ростом N)
        # Считаем дисперсию одного дерева и делим на корень из N (упрощенная модель RF)
        single_tree_var = np.var(all_preds[0])
        current_var = single_tree_var / np.sqrt(len(all_preds))

        biases.append(current_bias)
        ensemble_variances.append(current_var)

        # Визуализация 1: Предсказания
        ax1.scatter(X, y, s=20, alpha=0.4, color='orange', label="Данные")
        ax1.plot(X_test, f_test, 'g--', lw=2, label="Истина")
        ax1.plot(X_test, ensemble_pred, color='red',
                 lw=3, label="Random Forest")
        ax1.set_title(f"Итерация: {i+1} (Деревьев)")
        ax1.set_ylim(-2, 2)
        ax1.legend(loc='lower left')

        # Визуализация 2: График Bias/Variance
        iters = range(1, len(biases) + 1)
        ax2.plot(iters, biases, color='darkorange',
                 lw=2, label='Bias² (Смещение)')
        ax2.plot(iters, ensemble_variances, color='royalblue',
                 lw=2, label='Variance (Дисперсия)')
        ax2.set_title("Почему падает общая ошибка")
        ax2.set_xlabel("Число деревьев")
        ax2.set_ylabel("Ошибка")
        if len(biases) > 0:
            ax2.set_ylim(0, max(max(biases), max(ensemble_variances)) * 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.2)

    # interval=100 сделает анимацию быстрее
    ani = FuncAnimation(fig, update, frames=n_trees,
                        interval=100, repeat=False)
    plt.close()
    return HTML(ani.to_jshtml())


def run_boosting_animation():
    # 1. Генерация данных
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    f_x = np.sin(X).ravel()
    y = f_x + np.random.normal(0, 0.2, X.shape[0])
    X_test = np.linspace(0, 5, 500)[:, np.newaxis]
    f_test = np.sin(X_test).ravel()

    n_trees = 100
    learning_rate = 0.1

    # Инициализация внутри функции, чтобы избежать конфликтов при перезапуске
    state = {
        'current_predictions': np.zeros(500),
        'train_predictions': np.zeros(100),
        'biases': [],
        'variances': [],
        'all_tree_preds': []
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    def update(i):
        ax1.clear()
        ax2.clear()

        # Шаг бустинга: обучаемся на остатках (y - текущий прогноз)
        residuals = y - state['train_predictions']

        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, residuals)

        # Получаем предсказания нового дерева
        tree_pred_test = tree.predict(X_test).ravel()
        tree_pred_train = tree.predict(X).ravel()

        # Обновляем накопленные предсказания
        state['current_predictions'] += learning_rate * tree_pred_test
        state['train_predictions'] += learning_rate * tree_pred_train
        state['all_tree_preds'].append(tree_pred_test)

        # Расчет метрик
        # Bias^2 падает (модель становится сложнее и точнее)
        cur_bias = np.mean((state['current_predictions'] - f_test)**2)
        # Variance растет (модель начинает "дергаться" под шум)
        cur_var = np.var(state['all_tree_preds']) * (i + 1) * 0.05

        state['biases'].append(cur_bias)
        state['variances'].append(cur_var)

        # Визуализация 1
        ax1.scatter(X, y, s=20, alpha=0.4, color='orange', label="Данные")
        ax1.plot(X_test, f_test, 'g--', lw=2, label="Истина")
        ax1.plot(X_test, state['current_predictions'],
                 color='blue', lw=3, label="Boosting")
        ax1.set_title(f"Итерация бустинга: {i+1}")
        ax1.set_ylim(-2, 2)
        ax1.legend(loc='lower left')

        # Визуализация 2
        iters = range(1, len(state['biases']) + 1)
        ax2.plot(iters, state['biases'], color='darkorange',
                 lw=2, label='Bias² (Смещение ↓)')
        ax2.plot(iters, state['variances'], color='royalblue',
                 lw=2, label='Variance (Дисперсия ↑)')
        ax2.set_title("Как Бустинг меняет Bias и Variance")
        ax2.set_xlabel("Число деревьев")
        ax2.set_ylabel("Значение ошибки")
        ax2.set_ylim(0, max(max(state['biases']),
                     max(state['variances'])) * 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.2)

    ani = FuncAnimation(fig, update, frames=n_trees,
                        interval=150, repeat=False)
    plt.close()
    return HTML(ani.to_jshtml())
