# numpy 从已有的数组创建数组

import numpy as np
#list
x=[1,2,3]
y=np.asarray(x,dtype=int)
print(x,'\n',y)
#tuple
x=(1,2,3)
y=np.asarray(x,dtype=float)
print(x,'\n',y)
#元组列表
x=[(1,2,3),(4,5)]
y=np.asarray(x)
print(y)