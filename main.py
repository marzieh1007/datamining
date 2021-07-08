# import package

import pandas as pd
from pandas.api.types import is_numeric_dtype
# import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import math
import random
import numpy as np
import datetime
# from pprint import pprint as p
import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import pairwise_distances


# ====================================================================================================

def Elbow_algorithm(iris_data):
    time_start = datetime.datetime.now()
    x = iris_data.iloc[:, [2, 3]].values
    sse = []
    for k in range(1, 10):
        model = KMeans(n_clusters=k, random_state=1)
        model.fit(x)
        sse.append(model.inertia_)

    plt.plot(range(1, 10), sse)
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.show()

    model = KMeans(n_clusters=2, random_state=1)
    model.fit(x)
    y = model.labels_
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c="red", label="cluster 0")
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c="blue", label="cluster 1")
    plt.legend()
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", s=100)
    plt.show()

    time_finish = datetime.datetime.now()
    time_duration = time_finish - time_start
    # print('Execution Time Elbow_algorithm', int(time_duration.total_seconds()))
    print('Execution Time Elbow_algorithm: ', time_duration)


# ====================================================================================================

def Gap_Statistic_algorithm(iris_data):
    x = iris_data.iloc[:, [2, 3]].values


# =====================================================================================================

def Silhouette_Coefficient_algorithm(iris_data):
    time_start = datetime.datetime.now()
    x = iris_data.iloc[:, [2, 3]].values
    color = ['red', 'blue', 'green', 'yellow', 'aqua', 'fuchsia']
    score = []

    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=1)
        model.fit(x)
        score.append(silhouette_score(x, model.labels_))
        y = model.labels_
        for i in range(0, k):
            plt.scatter(x[y == i, 0], x[y == i, 1], c=color[i], label=i)
        plt.legend()
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", s=100)
        plt.show()
        # print('score: ', score)

    time_finish = datetime.datetime.now()
    time_duration = time_finish - time_start
    # print('Execution Time Silhouette_Coefficient_algorithm', int(time_duration.total_seconds()))
    print('Execution Time Silhouette_Coefficient_algorithm: ', time_duration)


# =====================================================================================================
def Canopy_algorithm(iris_data):
    time_start = datetime.datetime.now()
    x = iris_data.iloc[:, [2, 3]].values
    x_copy = iris_data.iloc[:, [2, 3]].values
    t1 = 2.0
    t2 = 1.8
    canopies = []
    while len(x) != 0:
        rand_index = random.randint(0, len(x) - 1)
        current_center = x[rand_index]
        current_center_list = []
        delete_list = []
        x = np.delete(x, rand_index, 0)
        for datum_j in range(len(x)):
            datum = x[datum_j]
            distance = math.sqrt(((current_center - datum) ** 2).sum())
            if distance < t1:
                current_center_list.append(datum)
            if distance < t2:
                delete_list.append(datum_j)
        x = np.delete(x, delete_list, 0)
        canopies.append((current_center, current_center_list))

    # print('canopies: ', canopies)
    fig = plt.figure()
    sc = fig.add_subplot(111)
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'cyan', 'pink', 'Violet', 'Fuchsia', 'Purple', 'Lime',
              'Olive', 'Brown', 'Gray', 'gold', 'orchid']
    markers = ['*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+', '1', '2', '^',
               '<', '>', '.', '4', 'H', '+', '1', '2', 's', 'p', 'x', 'D', 'd', '|', '_']
    for i in range(len(canopies)):
        canopy = canopies[i]
        center = canopy[0]
        components = canopy[1]
        sc.plot(center[0], center[1], marker=markers[i], color=colors[i], markersize=10)
        t1_circle = plt.Circle(xy=(center[0], center[1]), radius=t1, color='dodgerblue', fill=False)
        t2_circle = plt.Circle(xy=(center[0], center[1]), radius=t2, color='skyblue', alpha=0.2)
        sc.add_artist(t1_circle)
        sc.add_artist(t2_circle)
        for component in components:
            sc.plot(component[0], component[1], marker=markers[i], color=colors[i], markersize=1.5)
    maxvalue = np.amax(x_copy)
    minvalue = np.amin(x_copy)
    plt.xlim(minvalue - t1, maxvalue + t1)
    plt.ylim(minvalue - t1, maxvalue + t1)
    plt.show()

    time_finish = datetime.datetime.now()
    time_duration = time_finish - time_start
    # print('Execution Time Canopy_algorithm', int(time_duration.total_seconds()))
    print('Execution Time Canopy_algorithm: ', time_duration)


# =====================================================================================================

# read data
iris_data = pd.read_csv(r"C:\Users\MYava\Desktop\data set\iris_csv.csv")
iris_data.columns = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']

print("--------------------------------------------------------------------------------")

Elbow_algorithm(iris_data)

print("--------------------------------------------------------------------------------")

# Gap_Statistic_algorithm(iris_data)

print("--------------------------------------------------------------------------------")

Silhouette_Coefficient_algorithm(iris_data)

print("--------------------------------------------------------------------------------")

Canopy_algorithm(iris_data)

print("--------------------------------------------------------------------------------")

# =====================================================================================================

# show Min, Max, Mean, Mode, Median
for col in iris_data.columns:
    if is_numeric_dtype(iris_data[col]):
        print('%s:' % (col))
        print('Min = %.2f' % iris_data[col].min())
        print('Max = %.2f' % iris_data[col].max())
        print('Mean = %.2f' % iris_data[col].mean())
        print('Mode = %.2f' % iris_data[col].mode())
        print('Median = %.2f' % iris_data[col].median())
        print('Q1 = %.2f' % iris_data[col].quantile(0.25))
        print('Q3 = %.2f' % iris_data[col].quantile(0.75))
        print('IQR = %.2f' % (iris_data[col].quantile(0.75) - iris_data[col].quantile(0.25)))
        print("--------------------------------------------------------------------------------")

# show outlier
for col in iris_data.columns:
    if is_numeric_dtype(iris_data[col]):
        Selected_column = iris_data[col]
        print('%s:' % (col))
        Q1 = Selected_column.quantile(0.25)
        Q3 = Selected_column.quantile(0.75)
        IQR = Q3 - Q1
        Min = Selected_column.min()
        if Min <= (Q1 - 1.5 * IQR):
            Min = (Q1 - 1.5 * IQR)
        Max = Selected_column.max()
        if Max > (Q3 + 1.5 * IQR):
            Max = (Q3 + 1.5 * IQR)
        Outlier = Selected_column[(Selected_column < Min) | (Selected_column > Max)]
        print('outlier: ', Outlier)
        print("--------------------------------------------------------------------------------")

# show box plot for numeric data
sb.boxplot(data=iris_data)
plt.show()

iris_data.boxplot()
plt.show()

# show bar graph for nominal data
sb.countplot(x='class', data=iris_data)
plt.show()

# Parallel Coordinates
parallel_coordinates(iris_data, "class")
plt.ioff()

# scatter
sb.set_style("whitegrid")
sb.FacetGrid(iris_data, hue="class", height=6).map(plt.scatter, 'sepallength', 'sepalwidth').add_legend()
plt.show()

sb.set_style("whitegrid")
sb.FacetGrid(iris_data, hue="class", height=6).map(plt.scatter, 'petallength', 'petalwidth').add_legend()
plt.show()

# histogram sepal length
plt.figure(figsize=(10, 7))
x = iris_data["sepallength"]
plt.hist(x, bins=20, color="blue")
plt.title("Sepal Length in cm")
plt.xlabel("sepallength")
plt.ylabel("Count")
plt.show()

# histogram sepal width
plt.figure(figsize=(10, 7))
x = iris_data["sepalwidth"]
plt.hist(x, bins=20, color="orange")
plt.title("Sepal Width in cm")
plt.xlabel("sepalwidth")
plt.ylabel("Count")
plt.show()

# histogram petal length
plt.figure(figsize=(10, 7))
x = iris_data["petallength"]
plt.hist(x, bins=20, color="green")
plt.title("Petal Length in cm")
plt.xlabel("petallength")
plt.ylabel("Count")
plt.show()

# histogram petal width
plt.figure(figsize=(10, 7))
x = iris_data["petalwidth"]
plt.hist(x, bins=20, color="red")
plt.title("Petal Width in cm")
plt.xlabel("petalwidth")
plt.ylabel("Count")
plt.show()

iris_data = pd.read_csv(r"C:\Users\MYava\Desktop\data set\iris_csv.csv")
iris_data.columns = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
