import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
reg=linear_model.LinearRegression()

climate_C=pd.read_csv("./2019E/en_climate_monthly_C_3012208_1937-2005_P1M.csv")
climate_E=pd.read_csv("./2019E/en_climate_monthly_E_8403506_1942-2012_P1M.csv")
climate_N=pd.read_csv("./2019E/en_climate_monthly_N_2400300_1950-2006_P1M.csv")
climate_SW=pd.read_csv("./2019E/en_climate_monthly_SW_1107H7R_1980-1987_P1M.csv")
# figure1
temp_C=pd.DataFrame([climate_E['Date/Time'],climate_E['Mean Temp (°C)']])
temp_C=temp_C.dropna(axis=1)
plt.subplot(1,1,1)
plt.plot(temp_C.loc['Mean Temp (°C)'])
plt.xticks(np.arange(0,800,120),np.arange(1937,1998,10))
plt.title('Mean Temp (°C)')

reg_temp=pd.DataFrame(temp_C.values.T, index=temp_C.columns, columns=temp_C.index)
x=np.asarray([reg_temp.index,reg_temp.index]).T
y=np.asarray(reg_temp['Mean Temp (°C)'].values)
reg.fit(x,y)

y_p=reg.predict(x[0:])
plt.plot(x[0:],y_p,'--r')
plt.grid(True)
plt.show()



