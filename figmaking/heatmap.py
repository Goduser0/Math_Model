import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import palettable #python颜色库
from sklearn import datasets 

plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False 

iris=datasets.load_iris()
iris_data,iris_target=iris.data,iris.target

df=pd.DataFrame(np.hstack((iris_data,iris_target.reshape(150,1))),columns=np.append(iris.feature_names,'class'))

