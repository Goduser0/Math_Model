from re import X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 载入中文字体
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# 设定画布大小
plt.figure(dpi=200)

# 设定总标题
plt.suptitle('总标题',
             x=0.5,# x轴位置
             y=0.98,# y轴位置
             size=15,# 字体大小
             ha='center',# 水平对齐方式
             va='top',# 垂直对齐方式
             weight='bold',# 字体粗细
             rotation=0,# 旋转角度
             )
# 设定子图数量
plt.subplot(2,1,1)

# 设定自定义字体 fontdict关键字选用
font_self1={'family':'Microsoft YaHei','fontsize':12,'fontweight':'bold','color':(.01,.99,.99)}

# 子图标题
plt.title('fig1 title')

# 子图框线显示
plt.gca().spines['top'].set_visible(False) # top/bottom/left/right
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_linestyle('--')

# 网格线显示
plt.grid(False)# axis='x'

#坐标轴刻度（tick）与刻度值（tick label）操作
plt.tick_params(axis='both',#对那个方向（x方向：上下轴；y方向：左右轴）的坐标轴上的tick操作，可选参数{'x', 'y', 'both'}
                which='both',#对主刻度还是次要刻度操作，可选参数为{'major', 'minor', 'both'}
                colors='r',#刻度颜色
                
                #以下四个参数控制上下左右四个轴的刻度的关闭和开启
                top='on',#上轴开启了刻度值和轴之间的线
                bottom='on',#x轴关闭了刻度值和轴之间的线
                left='on',
                right='on',
                
                direction='out',#tick的方向，可选参数{'in', 'out', 'inout'}                
                length=10,#tick长度
                width=2,#tick的宽度
                pad=10,#tick与刻度值之间的距离
                labelsize=10,#刻度值大小
                labelcolor='#008856',#刻度值的颜色
                zorder=0,
                
                #以下四个参数控制上下左右四个轴的刻度值的关闭和开启
                labeltop='on',#上轴的刻度值也打开了此时
                labelbottom='on',                
                labelleft='on',
                labelright='off',
                
                labelrotation=45,#刻度值与坐标轴旋转一定角度
                
                grid_color='pink',#网格线的颜色，网格线与轴刻度值对应，前提是plt.grid()开启了
                grid_alpha=1,#网格线透明度
                grid_linewidth=10,#网格线宽度
                grid_linestyle='-',#网格线线型，{'-', '--', '-.', ':', '',matplotlib.lines.Line2D中的都可以用              
                
                
               )

plt.xlabel('fig1 xlabel')

plt.ylabel('fig1 ylabel')
# plt.xticks()
# plt.yticks()

# 
fig1_x=np.asarray([1,2,3])
fig1_y=np.asarray([1,2,3])
plt.plot(fig1_x,fig1_y)
plt.legend()
plt.show()
