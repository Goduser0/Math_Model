import pandas as pd
import numpy as np
a = np.array([[1, 2], [4, 5], [7, 8]], dtype=complex)
print(a)
b = a.reshape((2, 3))
print(a, '\n', b)

c = np.empty([2, 3], dtype=int, order='C')
print(c)

d = np.zeros(5)
print(d)

e = np.zeros([3, 2])
print(e[0])

