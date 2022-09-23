# 散点图

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
iris=datasets.load_iris()
x,y=iris.data,iris.target
pd_iris=pd.DataFrame(np.hstack((x,y.reshape(150,1))),columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)','class'])

plt.figure(dpi=150)
plt.scatter(pd_iris['sepal length (cm)'],pd_iris['sepal width (cm)'])
plt.show()
