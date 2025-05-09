import numpy as np
import math
from dataset2 import ti, yi
#------------------------------------------------------
def polinomIO(t,x):
    yhat = []
    for ti in t:
        toplam = 0
        for i in range(0,len(x)):
            toplam += x[i]*ti**i
            yhat.append(toplam)
    return yhat
#------------------------------------------------------
def findx(ti,yi,polinomderecesi):
    numofdata = len(ti)
    J = -np.ones((numofdata,1))
    for n in range(1,polinomderecesi+1):
        J = np.hstack((J,-np.ones((numofdata,1))*np.array(ti).reshape(numofdata,1)**n))
    A = np.linalg.inv(J.transpose().dot(J))
    B = J.transpose().dot(yi)
    x = -A.dot(B)
    return x
#------------------------------------------------------
def plotresult(ti,yi,x,fvalidation):
    import matplotlib.pyplot as plt
    T = np.arange(min(ti),max(ti),0.1)
    yhat = polinomIO(T,x)
    plt.scatter(ti,yi,color='darkred', marker='x')
    plt.plot(T,yhat,color = 'green', linestyle = 'solid',linewidth = 1)
    plt.xlabel('ti')
    plt.ylabel('yi')
    plt.title(str(len(x)-1)+'.- dereceden polinom modeli / FV:'+str(fvalidation),fontstyle='italic')
    plt.grid(color='green',linestyle='--',linewidth = 0.1)
    plt.legend(['polinom modeli','gercek veri'])
    plt.show()
#------------------------------------------------------
trainingindices = np.arenge(0,len(ti),2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1,len(ti),2)
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]


PD = []; FV = []
for polinomderecesi in range(1,10):
    x = findx(traininginput,trainingoutput,polinomderecesi)
    yhat = polinomIO(validationinput, x)
    e = np.array(validationoutput) - np.array(yhat)
    fvalidation = sum(e**2)
    PD.append(polinomderecesi)
    FV.append(fvalidation)
    print(polinomderecesi,fvalidation)
    plotresult(ti,yi,x,fvalidation)
#------------------------------------------------------
import mathplotlib.pyplot as plt
plt.bar(PD,FV,color = 'darkred')
plt.xlabel('polinom derecesi')
plt.ylabel('validation performansi')
plt.title('Polinom Modeli',fontstyle='italic')
plt.grid(color = 'green',linestyle='--',linewidth=0.1)
plt.show()
#------------------------------------------------------