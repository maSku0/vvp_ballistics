import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from classes.projectile import Projectile

import constant as const


def trajectoryODE(projectile,v=300,alpha0=0,Cd=0.25):
    def dragForce(v_vec):
        return -0.5 * const.RHO * A * Cd * np.linalg.norm(v_vec)**2 * (v_vec/np.linalg.norm(v_vec))

    def deriv(state, t, Cd, A, m):
        x, y, vx, vy = state
        
        v_vec = np.array([vx, vy])
        v = np.linalg.norm(v_vec)

        F_drag = dragForce(v_vec)

        dxdt = vx
        dydt = vy
        dvxdt = F_drag[0] / m
        dvydt = F_drag[1] / m - const.G

        return [dxdt, dydt, dvxdt, dvydt]

    x0 = 0.0
    y0 = 0.0
    initial_state = [x0, y0, v*np.cos(alpha0), v*np.sin(alpha0)]

    A = projectile.A
    m = projectile.m_kg

    t = tLinspace()

    trajectory = odeint(deriv, initial_state, t, args=(Cd, A, m))
    return trajectory

def tLinspace():
    return np.linspace(0, 10, 1000)

def createTable(values, ran=500, step=100):
    dim = len(values) #3
    leng = int(ran/step) #5
    res = np.zeros((leng,dim))
    res[:,0] = np.arange(step,ran+step,step)
    for x in range(leng):
        for i in values[0]:
            if i >= res[x,0]:
                for val in range(1,dim):
                    res[x,val] = float(values[val][values[0].index(i)])

                break
    return res

def compact(res, ran=500, step=100):
    rows,cols = res.shape
    new = np.zeros((int(ran/step),cols))
    x = np.arange(step,ran+step,step)
    for xa in range(len(x)):
        for i in range(rows):
            if res[i,0]>x[xa]:
                #write
                new[xa,0] = x[xa]
                for y in range(1,cols):
                    new[xa,y] = res[i,y]
                break
    return new

def normTwoCols(res,tup):
    res[:,tup[0]] = np.sqrt(res[:, tup[0]].copy()[:]**2+res[:, tup[1]].copy()[:]**2)
    return np.delete(res.copy(),tup[1],1)

def printTable(res, names=("x","y","v"), compacted=False, ran=500, step=100, normalise=False, cols=(0,0)):
    if(compacted):
        res = compact(res,ran,step)
    if(normalise):
        if(np.linalg.norm(cols)==0):
            raise Exception("Collumns for normalisation not specified!")
        else:
            res = normTwoCols(res,cols)
    for i in range(len(res)):
        print(f"{names[0]}: {res[i][0]}",end="")
        for y in range(1,len(res[i])):
            if(y<len(names)): print(f", {names[y]}: {res[i][y]}",end="")
            else: print(f", {res[i][y]}",end="")
        print("")

def approximate(table, deg=2):
    koeficienty = np.polyfit(table[:,0], table[:,1], deg)
    return np.poly1d(koeficienty)

def approximateCustom(table,i_1,i_2, deg=2):
    koeficienty = np.polyfit(table[:,i_1], table[:,i_2], deg)
    return np.poly1d(koeficienty)

def plot(x,y,name='Ballistic Trajectory with Drag',x_label='Horizontal Distance (m)',y_label='Vertical Distance (m)',x_limit=(0,0),y_limit=(0,0)):
    plt.plot(x, y)
    plt.title(name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if(np.linalg.norm(x_limit)!=0): plt.xlim(x_limit[0],x_limit[1]) #0,400
    if(np.linalg.norm(x_limit)!=0): plt.ylim(y_limit[1],y_limit[0]) #-5,1
    plt.show()
    
def compare(x,y,x2,y2,name='Comparison of two trajectories',x_label='Horizontal Distance (m)',y_label='Vertical Distance (m)',x_limit=(0,0),y_limit=(0,0),legend=True,name1='1',name2='2'):
    plt.title(name)
    plt.plot(x, y, color='blue',label=name1)
    plt.plot(x2,y2, color='red',label=name2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if(legend):plt.legend()
    if(np.linalg.norm(x_limit)!=0): plt.xlim(x_limit[0],x_limit[1]) #0,400
    if(np.linalg.norm(x_limit)!=0): plt.ylim(y_limit[1],y_limit[0]) #-5,1
    plt.show()