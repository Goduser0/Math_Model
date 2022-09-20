#numpy迭代数组

import numpy as np

a=np.arange(6).reshape(2,3)
print('原始数据：','\n',a)
#np.nditer为迭代器
#order='F'/'C' 列/行优先 缺省时为行优先

a1=np.copy(a,order='C')
print('a1',a1)
for i in np.nditer(a1):
    print(i)
a2=np.copy(a,order='F')
print('a2',a2)
for i in np.nditer(a2):
    print(i)

print('行优先')
for i in np.nditer(a,order='F'):
    print(i)
print('列优先')
for i in np.nditer(a,order='C'):
    print(i)

