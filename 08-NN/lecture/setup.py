
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.datasets import load_iris

def run_animate(func):
    # 1. Подготовка данных (2 признака для 2D визуализации)
    iris = load_iris()
    X = torch.FloatTensor(iris.data[:, 2:])
    y = torch.LongTensor(iris.target)

    # 2. Модель и оптимизатор
    model = nn.Sequential(nn.Linear(2, 10), func(), nn.Linear(10, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    # Сетка для отрисовки фона
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(epoch):
        ax.clear()

        # Шаг обучения (делаем 5 итераций на 1 кадр для скорости)
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # Предсказание для фона
        with torch.no_grad():
            Z = model(grid_tensor).argmax(1).reshape(xx.shape)

        # Отрисовка
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.brg)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.brg)
        ax.set_title(f"Эпоха: {epoch * 5} | Loss: {loss.item():.4f}")
        ax.set_xlabel('Petal length')
        ax.set_ylabel('Petal width')

    # Создание анимации
    ani = FuncAnimation(fig, update, frames=40, interval=100)
    plt.close() # Чтобы не дублировать статический график

    # Отображение
    return HTML(ani.to_jshtml())

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.datasets import load_iris

def run_animate(func):
    # 1. Подготовка данных (2 признака для 2D визуализации)
    iris = load_iris()
    X = torch.FloatTensor(iris.data[:, 2:])
    y = torch.LongTensor(iris.target)

    # 2. Модель и оптимизатор
    model = nn.Sequential(nn.Linear(2, 10), func(), nn.Linear(10, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    # Сетка для отрисовки фона
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(epoch):
        ax.clear()

        # Шаг обучения (делаем 5 итераций на 1 кадр для скорости)
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # Предсказание для фона
        with torch.no_grad():
            Z = model(grid_tensor).argmax(1).reshape(xx.shape)

        # Отрисовка
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.brg)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.brg)
        ax.set_title(f"Эпоха: {epoch * 5} | Loss: {loss.item():.4f}")
        ax.set_xlabel('Petal length')
        ax.set_ylabel('Petal width')

    # Создание анимации
    ani = FuncAnimation(fig, update, frames=40, interval=100)
    plt.close() # Чтобы не дублировать статический график

    # Отображение
    return HTML(ani.to_jshtml())
