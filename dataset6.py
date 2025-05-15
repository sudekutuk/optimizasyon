import numpy as np
import math

ti = np.arange(-20,20,0.27)
yi = [math.sin(t)/(t)+np.random.random()*0.1 for t in ti]
#------------------------------------------------------------
import matplotlib.pyplot as plt
plt.scatter(ti,yi,color='darkred',s=1)
plt.xlabel('ti')
plt.ylabel('yi')
plt.title('Dataset 6',fontstyle = 'italic')
plt.grid(color = 'green',linestyle = '--',linewidth = 0.1)
plt.show()
#=------------------------------------------------------------