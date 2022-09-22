import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#设定绘图数量
plt.subplot(1,1,1)
fig1_x=np.asarray([1,2,3])
fig1_y=np.asarray([1,2,3])
plt.plot(fig1_x,fig1_y)
plt.title('fig1 title')
plt.xlabel('fig1 xlabel')
plt.ylabel('fig1 ylabel')
# plt.xticks()
# plt.yticks()
plt.legend()
plt.show()
