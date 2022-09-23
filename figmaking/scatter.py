# 散点图

from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

iris=datasets.load_iris()
x,y=iris.data,iris.target
pd_iris=pd.DataFrame(np.hstack((x,y.reshape(150,1))),columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)','class'])
print(type(iris))

plt.figure(dpi=150)
iris_type=pd_iris['class'].unique()
i_name=iris.target_names
i_marker=['.',',','$\clubsuit$']
i_color=['#c72e29','#098154','#fb832d']

for i in range(len(iris_type)):
    plt.scatter(
                pd_iris.loc[pd_iris['class']==iris_type[i],'sepal length (cm)'],
                pd_iris.loc[pd_iris['class']==iris_type[i],'sepal width (cm)'],
                s=50,
                color=i_color[i],
                alpha=0.8,
                # facecolor='r',
                # edgecolors='none',
                marker=i_marker[i],
                linewidths=1,
                label=i_name[i]
                )
plt.legend()
plt.show()


