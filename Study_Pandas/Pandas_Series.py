import pandas as pd
import numpy as np

# pandas.Series( data, index, dtype, name, copy)
# data：一组数据(ndarray 类型)。
# index：数据索引标签，如果不指定，默认从 0 开始。
# dtype：数据类型，默认会自己判断。
# name：设置名称。
# copy：拷贝数据，默认为 False。
a=np.arange(1,5)
b=pd.Series(a)
print(b)

b1=pd.Series(a,index=['x','y','z','q'])
print(b1)
print(b1['y'])


