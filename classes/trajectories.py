from types import NoneType
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from .projectile import Projectile 

from .constant import *


'''
Calculation of trajectory for known projectile and speed
returns matrix in format [x,y,vx,vy]

Type of drag calculation can be changed by specifiing dragSource 
    - poly1d approximation of drag force at speed 
    - ndarray of two collumns, force and speed
'''

def trajectory(projectile:Projectile,v:float=300,alpha0:float=0,dragSource:np.poly1d|np.ndarray|None=None)->np.ndarray:
    def dragForce(v_vec):
        F = -0.5 * RHO * A * Cd * np.linalg.norm(v_vec)**2 * (v_vec/np.linalg.norm(v_vec))
        return F
    
    def dragForceMatrix(v_vec):
        v_norm = np.linalg.norm(v_vec)
        for i in range(len(dragSource[:,0])):
            if(v_norm < dragSource[i,0]):
                F = np.abs(dragSource[i,1]*(v_vec/v_norm))
                F[F<0] = 0
                return -1*F
            else:
                return -1*np.abs(dragSource[i,1]*(v_vec/v_norm))
            
    def dragForcePolynomial(v_vec):
        v_norm = np.linalg.norm(v_vec)
        return dragSource(v_norm)*(v_vec/v_norm)
            
    def deriv(state, t, Cd, A, m):
        x, y, vx, vy = state
        
        v_vec = np.array([vx, vy])
        v = np.linalg.norm(v_vec)

        F_drag = np.array([0,0])

        if(type(dragSource)==NoneType):
            F_drag = dragForce(v_vec)
        elif(type(dragSource)==np.poly1d):
            F_drag = dragForcePolynomial(v_vec)
        elif(type(dragSource)==np.ndarray):
            F_drag = dragForceMatrix(v_vec)
        
        dxdt = vx
        dydt = vy
        dvxdt = F_drag[0] / m
        dvydt = F_drag[1] / m - G
        
        return [dxdt, dydt, dvxdt, dvydt]

    x0 = 0.0
    y0 = 0.0
    initial_state = [x0, y0, v*np.cos(alpha0), v*np.sin(alpha0)]

    A = projectile.A
    m = projectile.m_kg
    Cd = projectile.G1

    t = tLinspace()

    trajectory = odeint(deriv, initial_state, t, args=(Cd, A, m))
    return trajectory

'''
returns stock time linspace from 0 to 10 seconds
'''
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

'''
Inputs slice of speed and time [v,t] in collumns
returns polynomial with speed input and drag force output

output drag force is for 1kg projectile, unless specified with "projectile" input
whitch returns drag force at speeds relevant specifically for that projectile

function can also norm two collumns with norm=True flag
preset for formating is set at collumns 0 and 1, it can be changed by changing "norm_at=(0,1)" to desired collumns
'''
def createDragFunction(slice_v_t, norm=False, norm_at=(0,1), projectile:Projectile|None=None, deg=2):
    if(norm): slice_v_t = normTwoCols(slice_v_t,norm_at)
    v_p = approximateCustom(slice_v_t,1,0,deg)
    v_pdt = v_p.deriv()
    F_dt = v_pdt
    if(projectile!=None):
        F_dt = v_pdt*projectile.m_kg
    koeficienty = np.polyfit(slice_v_t[:,0], F_dt(slice_v_t[:,1]), deg)
    return np.poly1d(koeficienty)

'''
inputs expanded matrix
returns compacted matrix for x ranges up to 500 meters (ran=) with steps of 100 (step=)
'''
def compact(res:np.ndarray, ran=500, step=100)->np.ndarray:
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

'''
returns matrix with two collumns (selected by tuple "tup") normalised

example:
res[1,3,4,6] --> normTwoCols(res,tup=(1,2)) -> res[1,5,6]
'''
def normTwoCols(res:np.ndarray,cols:tuple)->np.ndarray:
    res[:,cols[0]] = np.sqrt(res[:, cols[0]].copy()[:]**2+res[:, cols[1]].copy()[:]**2)
    return np.delete(res.copy(),cols[1],1)

'''
prints table defaultly named (x,y,v)
names can be specified (names=) for more or less collumn input
table can also be compacted (same as compact() method) with arguments compacted=True, ran=x, step=y
    and normalised (same as normTwoCols() method) with normalise=True, cols=(x,y)
'''
def printTable(res:np.ndarray, names=("x","y","v"), compacted=False, ran=500, step=100, normalise=False, cols=(0,0)):
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
'''
for x value (range) prints all remaining values from trajectory matrix
'''
def getValues(trajectory:np.ndarray, x:int)->np.ndarray:
    for i in range(len(trajectory[:,0])):
        if trajectory[i,0] > x:
            return trajectory[i,1:-1]
        
'''
returns poly1d approximation for table[x,y] -> p(x) = y of defaultly 2 degree (deg=)
'''
def approximate(table, deg=2):
    koeficienty = np.polyfit(table[:,0], table[:,1], deg)
    return np.poly1d(koeficienty)

'''
returns poly1d approximation for table[i_1,i_2] -> p(i_1) = i_2 of defaultly 2 degree (deg=)
with i_1 and i_2 as inputs
'''
def approximateCustom(table,i_1,i_2, deg=2):
    koeficienty = np.polyfit(table[:,i_1], table[:,i_2], deg)
    return np.poly1d(koeficienty)

'''
Returns expanded matrix by collumn of time

Use only with generated/uncompacted table
'''
def addTimeToSolutionMatrix(res):
    rows,cols=res.shape
    new = np.zeros((rows,cols+1))
    new[:,:-1] = res.copy()
    new[:,-1] = tLinspace().T
    return new

'''
Plots trajectory on graph
Inputs:
    trajectory - gets first and second collumn for x and y coordinates
        if trajectory is not specified it will look for parameters x= and y=
    name - Name of graph
    x_label/y_label - labels of x/y axis
    x_limit/y_limit - tuple of ranges for x/y
'''
def plot(x=None,y=None,name='Graph',x_label='x',y_label='y',x_limit=(0,0),y_limit=(0,0),trajectory=NoneType):
    if(type(trajectory) is np.ndarray):
        name='Ballistic Trajectory with Drag'
        x_label='Horizontal Distance (m)'
        y_label='Vertical Distance (m)'
        plt.plot(trajectory[:,0],trajectory[:,1])
    else: 
        plt.plot(x, y)
    plt.title(name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if(np.linalg.norm(x_limit)!=0): plt.xlim(x_limit[0],x_limit[1]) #0,400
    if(np.linalg.norm(y_limit)!=0): plt.ylim(y_limit[1],y_limit[0]) #-5,1
    plt.show()

'''
Plots two trajectories on graph as comparison
Inputs:
    trajectory1/trajectory2 - gets first and second collumns for x and y coordinates
        if trajectory is not specified it will look for parameters x1/x2= and y1/y2=
    name - Name of graph
    x_label/y_label - labels of x/y axis
    x_limit/y_limit - tuple of ranges for x/y
    legend - if True will print legend with names name1/name2=
''' 
def compare(x1=None,y1=None,x2=None,y2=None,name='Comparison of two trajectories',x_label='Horizontal Distance (m)',y_label='Vertical Distance (m)',x_limit=(0,0),y_limit=(0,0),legend=True,name1='1',name2='2',trajectory1=NoneType,trajectory2=NoneType):
    plt.title(name)
    if(type(trajectory1) is np.ndarray):
        plt.plot(trajectory1[:,0],trajectory1[:,1], color='blue',label=name1)
    else: 
        plt.plot(x1, y1, color='blue',label=name1)
    if(type(trajectory2) is np.ndarray):
        plt.plot(trajectory2[:,0],trajectory2[:,1], color='red',label=name2)
    else: 
        plt.plot(x2, y2, color='red',label=name2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if(legend):plt.legend()
    if(np.linalg.norm(x_limit)!=0): plt.xlim(x_limit[0],x_limit[1]) #0,400
    if(np.linalg.norm(x_limit)!=0): plt.ylim(y_limit[1],y_limit[0]) #-5,1
    plt.show()