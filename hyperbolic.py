import numpy as np
import math 
from dataset6 import ti,yi
#------------------------------------------------
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
#-------------------------------------------------
def hyperbolicIO(t,x):
    S = int((len(x)-1)/3)
    yhat = []
    for ti in t:
        toplam = x[3*S]
        for j in range(0,S):
            toplam += x[2*S+j]*tanh(x[j]*ti+x[S+j])
        yhat.append(toplam)
    return yhat
#-------------------------------------------------
def error(xk,ti,yi):
    yhat = hyperbolicIO(ti, xk)
    return np.array(yi) - np.array(yhat)
#-------------------------------------------------
def findJacobian(t,x):
    S = int((len(x)-1)/3)
    numofdata = len(t)
    J = np.matrix(np.zeros((numofdata,3*S+1)))
    for i in range(0,numofdata):
        for j in range(0,S):
            J[i,j] = -x[j+2*S]*t[i]*(1-tanh(x[j]*t[i]+x[S+j])**2)
        for j in range(0,2*S):
            J[i,j] = -x[j+S]*(1-tanh(x[j-S]*t[i]+x[j])**2)
        for j in range(2*S,3*S):
            J[i,j] = -tanh(x[j-2*S]*t[i]+x[j-S])
        J[i,3*S] = -1
    return J
#-------------------------------------------------
trainingindices = np.arenge(0,len(ti),2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1,len(ti),2)
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]
#-------------------------------------------------
MaxIter = 1000
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99
#----------------------------------------
S = 15 
xk = np.random.random(3*S+1)-0.5
k = 0; C1 = True; C2 = True; C3 = True;C4 = True; fvalidationBest = 1e99; kBest = 0

ek = error(xk,traininginput,traininginput)
ftraining = sum(ek**2)
FTRA = [ftraining]
evalidation = error(xk,validationinput,validationoutput)
fvalidation = sum(evalidation**2)
FVAL = [fvalidation]
ITERATION = [k]
print('k:',k,'f',format(ftraining,'f'))
mu = 1; muscal = 10; I = np.identity(3*S+1)
while C1 & C2 & C3 & C4:
    ek = error(xk,traininginput,traininginput)
    Jk = findJacobian(traininginput, trainingoutput)
    gk = np.array((2*Jk.transpose().dot(ek)).tolist()[0])
    Hk = 2*Jk.transpose().dot(Jk) + 1e-8*I
    ftraining = sum(ek**2)
    sk = 1
    loop = True
    while loop:
        zk = -np.linalg.inv(Hk+mu*I).dot(gk)
        zk = np.array(zk.tolist()[0])
        ez = error(xk+sk*zk,traininginput,trainingoutput)
        fz = sum(ez**2)
        if fz < ftraining:
            pk = 1*zk
            mu = mu/muscal
            k += 1
            xk = xk + sk*pk
            loop = False
            print('k:',k,'ftra:',format(fz,'f'),'fval',format(fvalidation,'f'),'fval*',format())


























