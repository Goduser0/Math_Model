import math
from collections.abc import Iterable
def quadratic(a,b,c):
    delta=(b**2)-(4*a*c)
    if delta>=0:
        delta=math.sqrt(delta)
    else:
        print('delta<0')
        return
    x1=(-b+delta)/(2*a)
    x2=(-b-delta)/(2*a)
    return x1,x2

print(quadratic(1,3,-4))
    