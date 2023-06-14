#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import scipy.stats as sts
from haversine import haversine
import matplotlib.pyplot as plt
import seaborn as sns

def plot_countplot(df, feature):
    """
    df: датафрейм, который содержит feature
    feature: исследуемый признак
    
    return: строит график распределения категориальной переменной
    """
    plt.figure()
    sns.countplot(y=df[feature])
    plt.title(f'Распределение переменной {feature}')
    plt.ylabel('Категории')
    plt.xlabel('Количество значений в категории')
    plt.show()
    
def plot_hist(df, feature):
    """
    df: датафрейм, который содержит feature
    feature: исследуемый признак
    
    return: строит гистограмму исследуемой переменной
    """
    plt.figure()
    plt.hist(df[feature], bins=60)
    plt.title(f'Распределение переменной {feature}')
    plt.xlabel('Значения')
    plt.ylabel('Количество попаданий в отрезок')
    plt.show()
    
def z_test_mu_diff_two_sided(x, y, alpha):
    """
    x: выборка №1
    y: выборка №2
    alpha: уровень значимости
    
    return: результат проверки двусторонней гипотезы о разности матожиданий
    """
    diff = x.mean() - y.mean() # разница выборочных средних
    nx, ny = x.size, y.size # размеры выборок
    diff_std = np.sqrt(x.var(ddof=1)/nx + y.var(ddof=1)/ny) # стандарное отклонение среднего
                                    
    z_obs = diff/diff_std # расчетная (наблюдаемая) статистика   
    z_crit = sts.norm.ppf(1 - alpha/2) # критическая статистика
    z_diff = np.absolute(z_obs) < z_crit # правда ли, что модуль расчетной меньше критической
    
    test_result = 'H0 не отвергается' if z_diff==True else 'H0 отвергается'
    return (test_result)

def z_test_mu_diff_right_sided(x, y, alpha):
    """
    x: выборка №1
    y: выборка №2
    alpha: уровень значимости
    
    return: результат проверки правосторонней гипотезы о разности матожиданий
    """
    diff = x.mean() - y.mean() # разница выборочных средних
    nx, ny = x.size, y.size # размеры выборок
    diff_std = np.sqrt(x.var(ddof=1)/nx + y.var(ddof=1)/ny) # стандарное отклонение среднего
                                    
    z_obs = diff/diff_std # расчетная (наблюдаемая) статистика   
    z_crit = sts.norm.ppf(1 - alpha) # критическая статистика
    z_diff = z_obs < z_crit # правда ли, что расчетная статистика меньше критической
    
    test_result = 'H0 не отвергается' if z_diff==True else 'H0 отвергается'
    return (test_result)

def manhattan_distance(df, lat1, lng1, lat2, lng2):
    """
    df: датафрейм, который содержит остальные аргументы функции
    lat1: широта объекта 1 (столбец)
    lng1: долгота объекта 1 (столбец)
    lat2: широта объекта 2 (столбец)
    lng2: долгота объекта 2 (столбец)
    
    return: возвращает манхэттенское расстояние между объектами 1 и 2
    """
    dists = []
    for i in range(len(df)):
        x = haversine((lat1[i], lng1[i]), (lat1[i], lng2[i]))
        y = haversine((lat1[i], lng1[i]), (lat2[i], lng1[i]))
        dist = x + y
        dists.append(dist)
    return dists

