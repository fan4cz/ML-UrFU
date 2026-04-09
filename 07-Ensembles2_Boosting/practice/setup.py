<<<<<<< HEAD

=======
>>>>>>> 02ba362e133e6c065d79e8433ee494ee786bfd4f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')


def animate_boosting_comparison(X, y, frames=30, interval=200):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    titles = ['Vanilla GBM', 'XGBoost', 'LightGBM', 'CatBoost']
    params = {'learning_rate': 0.1, 'random_state': 42}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.ravel()

    def update(frame):
        n_trees = frame + 1
        for i, title in enumerate(titles):
            ax = axes_flat[i]
            ax.clear()

            # Выбор модели
            if title == 'Vanilla GBM':
                model = GradientBoostingClassifier(
                    n_estimators=n_trees, **params)
            elif title == 'XGBoost':
                model = XGBClassifier(
                    n_estimators=n_trees, use_label_encoder=False, eval_metric='logloss', **params)
            elif title == 'LightGBM':
                model = LGBMClassifier(
                    n_estimators=n_trees, verbose=-1, **params)
            elif title == 'CatBoost':
                model = CatBoostClassifier(
                    n_estimators=n_trees, verbose=0, **params)

            model.fit(X, y)

            # Предсказание вероятностей для фона
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, cmap='RdBu', alpha=0.6)
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=30)
            ax.set_title(f"{title} | Trees: {n_trees}")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.close()

    return HTML(ani.to_jshtml())
<<<<<<< HEAD
=======

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
>>>>>>> 02ba362e133e6c065d79e8433ee494ee786bfd4f


warnings.filterwarnings('ignore')


def animate_boosting_comparison(X, y, frames=30, interval=200):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    titles = ['Vanilla GBM', 'XGBoost', 'LightGBM', 'CatBoost']
    params = {'learning_rate': 0.1, 'random_state': 42}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.ravel()

    def update(frame):
        n_trees = frame + 1
        for i, title in enumerate(titles):
            ax = axes_flat[i]
            ax.clear()

            # Выбор модели
            if title == 'Vanilla GBM':
                model = GradientBoostingClassifier(
                    n_estimators=n_trees, **params)
            elif title == 'XGBoost':
                model = XGBClassifier(
                    n_estimators=n_trees, use_label_encoder=False, eval_metric='logloss', **params)
            elif title == 'LightGBM':
                model = LGBMClassifier(
                    n_estimators=n_trees, verbose=-1, **params)
            elif title == 'CatBoost':
                model = CatBoostClassifier(
                    n_estimators=n_trees, verbose=0, **params)

            model.fit(X, y)

            # Предсказание вероятностей для фона
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, cmap='RdBu', alpha=0.6)
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=30)
            ax.set_title(f"{title} | Trees: {n_trees}")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.close()

    return HTML(ani.to_jshtml())
