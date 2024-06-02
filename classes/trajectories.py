from types import NoneType
from typing import Callable
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from .projectile import Projectile
from .constant import RHO, G, MACH


def trajectory(
        projectile: Projectile,
        v: float = 300,
        alpha0: float = 0,
        y0: float = 0,
        dragSource: np.poly1d | np.ndarray | None = None,
        Cd: float | None = None,
        dragType: str = "CUBIC",
        dragFunction: Callable = None
) -> np.ndarray:
    '''
    Calculation of trajectory for known projectile and speed
    returns matrix in format [x,y,vx,vy]

    Type of drag calculation can be changed by specifiing dragSource
        - poly1d approximation of drag force at speed
        - ndarray of two collumns, force and speed
    '''

    def dragForceMatrix(v_vec):
        v_norm = np.linalg.norm(v_vec)
        for i in range(len(dragSource[:, 0])):
            if (v_norm < dragSource[i, 0]):
                F = np.abs(dragSource[i, 1]*(v_vec/v_norm))
                F[F < 0] = 0
                return -1*F
            else:
                return -1*np.abs(dragSource[i, 1]*(v_vec/v_norm))

    def dragForcePolynomial(v_vec):
        v_norm = np.linalg.norm(v_vec)
        return dragSource(v_norm)*(v_vec/v_norm)

    def deriv(state, t, Cd, A, m):
        x, y, vx, vy = state
        v_vec = np.array([vx, vy])
        F_drag = np.array([0, 0])
        if (type(dragSource) is NoneType):
            if (dragType == "CUBIC"):
                F_drag = cubicDragFunction(v_vec, Cd, A)
            elif (dragType == "EXP"):
                F_drag = exponentialDragFunction(v_vec, Cd, A)
            elif (dragType == "CUSTOM"):
                F_drag = dragFunction(v_vec, Cd, A)
        elif (type(dragSource) is np.poly1d):
            F_drag = dragForcePolynomial(v_vec)
        elif (type(dragSource) is np.ndarray):
            F_drag = dragForceMatrix(v_vec)

        dxdt = vx
        dydt = vy
        dvxdt = F_drag[0] / m
        dvydt = F_drag[1] / m - G

        return [dxdt, dydt, dvxdt, dvydt]

    x0 = 0.0
    y0 = y0
    initial_state = [x0, y0, v*np.cos(alpha0), v*np.sin(alpha0)]

    A = projectile.A
    m = projectile.m_kg
    if (type(Cd) is NoneType):
        Cd = projectile.Cd
    else:
        Cd = Cd

    t = tLinspace()

    trajectory = odeint(deriv, initial_state, t, args=(Cd, A, m))
    return trajectory


def cubicDragFunction(
        v_vec: np.ndarray,
        Cd: float,
        A: float
) -> np.ndarray:
    '''
    Cubical function for for calculation with Cd
    '''
    v = np.linalg.norm(v_vec)
    F = -0.5 * RHO * A * Cd * (v**2) * (v_vec/v)
    return F


def exponentialDragFunction(
        v_vec: np.ndarray,
        Cd: float,
        A: float
) -> np.ndarray:
    '''
    Exponential function for calculation with Cd
    '''
    v = np.linalg.norm(v_vec)
    mach = v/MACH
    F = (np.e-1)**mach - 1
    F = F * (-1 * RHO * A * Cd)
    F = F * (v_vec/mach)
    return F*MACH


def approximateCd(
        projectile: Projectile,
        v0: float,
        v_s: np.ndarray,
        dragType: str = "CUBIC",
        dragFunction: Callable | None = None,
        min: int = 0,
        max: int = 500,
        step: int = 100,
        Cd0: float = 0.5
) -> float:
    '''
    Function for approximation of Cd with existing v_s matrix using drag
    function

    parameters:
    - projectile: projectile object with attributes `density`, `area`, `mass`
    - v0: base speed
    - v_s: matrix of measured values - speed at range
    - dragType: type of drag function
        - CUBIC
        - EXP
        - CUSTOM
            - requires `dragFunction` defined
            - `dragFunction` is a function with arguments:
                - v_vec: speed vector (speed in x and y)
                - Cd: drag coefficient
                - A: reference area
            - returns:
                - F: drag force in x and y
    '''
    min = v_s[0, 1]
    max = v_s[-1, 1]
    step = v_s[1, 1]

    def customDragFittingFunction(Cd_guess, projectile, v0):
        table = trajectory(projectile, v0, Cd=Cd_guess,
                           dragType=dragType, dragFunction=dragFunction)
        table = compact(table, max, step, min)
        table = normTwoCols(table, (2, 3))
        table = extractCollumns(table, (2))
        v = table.copy()
        error = np.sum((v - v_s[:, 1]) ** 2)
        return error

    result = minimize(customDragFittingFunction, Cd0, args=(projectile, v0))
    return result.x[0]


def tLinspace():
    '''
    returns stock time linspace from 0 to 10 seconds
    '''
    return np.linspace(0, 10, 1000)


def createDragFunction(
        table_v0_t1: np.ndarray,
        norm=False,
        norm_at=(0, 1),
        projectile: Projectile | None = None,
        deg=2
) -> np.poly1d:
    '''
    Inputs table of speed and time [v,t] in collumns
    returns polynomial with speed input and drag force output

    output drag force is for 1kg projectile, unless specified with
    "projectile" input
    whitch returns drag force at speeds relevant specifically for that
    projectile

    function can also norm two collumns with norm=True flag
    preset for formating is set at collumns 0 and 1, it can be changed by
    changing "norm_at=(0,1)" to desired collumns
    '''
    if (norm):
        table_v0_t1 = normTwoCols(table_v0_t1, norm_at)
    v_p = approximateCustom(table_v0_t1, 1, 0, deg)
    v_pdt = v_p.deriv()
    F_dt = v_pdt
    if (projectile is not None):
        F_dt = v_pdt*projectile.m_kg
    koeficienty = np.polyfit(table_v0_t1[:, 0], F_dt(table_v0_t1[:, 1]), deg)
    return np.poly1d(koeficienty)


def create_vs(
        v: tuple,
        s: tuple
) -> np.ndarray:
    '''
    inputs two tuples, speed and range
    returns v_s table (speed and range)
    '''
    v_s = np.zeros((len(s), 2))
    v_s[:, 0] = v
    v_s[:, 1] = s
    return v_s


def create_vt_from_vs(
        table_v0_s1: np.ndarray
) -> np.ndarray:
    '''
    inputs table of two collumns - speed and range
    returns table of speeds at times
    '''
    v_t = table_v0_s1.copy()
    v_t[:, 1] = table_v0_s1[:, 1]/table_v0_s1[:, 0]
    return v_t


def compact(
        table: np.ndarray,
        ran=500,
        step=100,
        min=100
) -> np.ndarray:
    '''
    inputs expanded matrix
    returns compacted matrix for x ranges up to 500 meters (ran=) with steps
    of 100 (step=)
    '''
    rows, cols = table.shape
    new = np.zeros((int((ran-min)//step) + 1, cols))
    x = np.arange(min, ran+step, step)

    for xa in range(len(x)):
        for i in range(rows):
            if table[i, 0] > x[xa]:
                new[xa, 0] = x[xa]
                for y in range(1, cols):
                    new[xa, y] = table[i, y]
                break
    return new


def normTwoCols(
        table: np.ndarray,
        cols: tuple
) -> np.ndarray:
    '''
    returns matrix with two collumns (selected by tuple "tup") normalised

    example:
    res[1,3,4,6] --> normTwoCols(res,tup=(1,2)) -> res[1,5,6]
    '''
    table[:, cols[0]] = np.sqrt(table[:, cols[0]].copy()[
                                :]**2+table[:, cols[1]].copy()[:]**2)
    return np.delete(table.copy(), cols[1], 1)


def printTable(
        table: np.ndarray,
        names: tuple | dict = ("x", "y", "v"),
        compacted=False,
        ran=500,
        step=100,
        normalise=False,
        cols=(0, 0),
        precision=5
) -> None:
    '''
    prints table defaultly named (x,y,v)

    names can be specified (names=) for more or less collumn input

    table can also be
        - compacted (same as compact() method) with arguments
        compacted=True, ran=x, step=y
        - and normalised (same as normTwoCols() method) with normalise=True,
        cols=(x,y)
    '''

    if (type(names) is tuple):
        names_d = dict((x, "") for x in names)
        names = names_d

    pref = list(names.keys())
    suf = list(names.values())
    if (compacted):
        table = compact(table, ran, step)
    if (normalise):
        if (np.linalg.norm(cols) == 0):
            raise Exception("Collumns for normalisation not specified!")
        else:
            table = normTwoCols(table, cols)
    for i in range(len(table)):
        for y in range(0, len(table[i])):
            value = table[i][y].copy()
            if (precision == 0):
                value = int(value)
            else:
                value = round(value, precision)
            if (y != 0):
                print(", ", end="")
            if (y < len(pref)):
                print(f"{pref[y]}: {value}{suf[y]}", end="")
            else:
                print(f", {value}", end="")
        print("")


def printDragFunction(
        f: Callable,
        Cd: float,
        A: float,
        min: int,
        max: int,
        vx0_vy1_table=NoneType
) -> None:
    '''
    helper function for plotting Drag function
    '''
    x = np.linspace(min, max)
    zer = np.zeros(x.shape)
    v_vec = np.array([x, zer]).T

    if (type(vx0_vy1_table) is np.ndarray):
        v_vec = vx0_vy1_table.copy()

    result = np.zeros_like(v_vec)

    for i in range(result.shape[0]):
        result[i, :] = f(v_vec[i, :], Cd, A)

    plot(v_vec[:, 0], -result[:, 0], x_label="v",
         y_label="F", name="V - F Graph")


def getValues(
        table_x0_values: np.ndarray,
        x: int
        ) -> np.ndarray:
    '''
    for x value (range) prints all remaining values from trajectory matrix
    '''
    for i in range(len(table_x0_values[:, 0])):
        if table_x0_values[i, 0] > x:
            return table_x0_values[i, 1:-1]


def approximate(
        table_2c: np.ndarray,
        deg=2
        ) -> np.poly1d:
    '''
    returns poly1d approximation for table[x,y] -> p(x) = y of defaultly 2
    degree (deg=)
    '''
    koeficienty = np.polyfit(table_2c[:, 0], table_2c[:, 1], deg)
    return np.poly1d(koeficienty)


def approximateCustom(
        table_2c: np.ndarray,
        i_1: int,
        i_2: int,
        deg: int = 2
        ) -> np.poly1d:
    '''
    returns poly1d approximation for table[i_1,i_2] -> p(i_1) = i_2 of
    defaultly 2 degree (deg=)
    with i_1 and i_2 as inputs
    '''
    koeficienty = np.polyfit(table_2c[:, i_1], table_2c[:, i_2], deg)
    return np.poly1d(koeficienty)


def addTimeToSolutionMatrix(
        table: np.ndarray
        ) -> np.ndarray:
    '''
    Returns expanded matrix by collumn of time

    Use only with generated/uncompacted table
    '''
    rows, cols = table.shape
    new = np.zeros((rows, cols+1))
    new[:, :-1] = table.copy()
    new[:, -1] = tLinspace().T
    return new


def extractCollumns(
        table: np.ndarray,
        cols=(0, 1)
        ) -> np.ndarray:
    '''
    returns matrix of selected collumns
    '''
    return table[:, cols]


def plot(
        x=None,
        y=None,
        name='Graph',
        x_label='x',
        y_label='y',
        x_limit=(0, 0),
        y_limit=(0, 0),
        table_x0_y1=NoneType
        ):
    '''
    Plots trajectory on graph
    Inputs:
        - table_x0_y1 - gets first and second collumn for x and y coordinates
            - if trajectory is not specified it will look for parameters x= and
            y=
        - name - Name of graph
        - x_label/y_label - labels of x/y axis
        - x_limit/y_limit - tuple of ranges for x/y
    '''
    if (type(table_x0_y1) is np.ndarray):
        name = 'Ballistic Trajectory with Drag'
        x_label = 'Horizontal Distance (m)'
        y_label = 'Vertical Distance (m)'
        plt.plot(table_x0_y1[:, 0], table_x0_y1[:, 1])
    else:
        plt.plot(x, y)
    plt.title(name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if (np.linalg.norm(x_limit) != 0):
        plt.xlim(x_limit[0], x_limit[1])  # 0,400
    if (np.linalg.norm(y_limit) != 0):
        plt.ylim(y_limit[1], y_limit[0])  # -5,1
    plt.show()


def compare(
        x1=None,
        y1=None,
        x2=None,
        y2=None,
        name='Comparison of two trajectories',
        x_label='Horizontal Distance (m)',
        y_label='Vertical Distance (m)',
        x_limit=(0, 0),
        y_limit=(0, 0),
        legend=True,
        name1='1',
        name2='2',
        table1_x0_y1=NoneType,
        table2_x0_y1=NoneType
        ) -> None:
    '''
    Plots two trajectories on graph as comparison
    Inputs:
        - table1_x0_y1/table2_x0_y1 - gets first and second collumns for x and
        y coordinates
            - if trajectory is not specified it will look for parameters
            x1/x2= and y1/y2=
        - name - Name of graph
        - x_label/y_label - labels of x/y axis
        - x_limit/y_limit - tuple of ranges for x/y
        - legend - if True will print legend with names name1/name2=
    '''
    plt.title(name)
    if (type(table1_x0_y1) is np.ndarray):
        plt.plot(table1_x0_y1[:, 0], table1_x0_y1[:, 1],
                 color='blue', label=name1)
    else:
        plt.plot(x1, y1, color='blue', label=name1)
    if (type(table2_x0_y1) is np.ndarray):
        plt.plot(table2_x0_y1[:, 0], table2_x0_y1[:, 1],
                 color='red', label=name2)
    else:
        plt.plot(x2, y2, color='red', label=name2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if (legend):
        plt.legend()
    if (np.linalg.norm(x_limit) != 0):
        plt.xlim(x_limit[0], x_limit[1])  # 0,400
    if (np.linalg.norm(x_limit) != 0):
        plt.ylim(y_limit[1], y_limit[0])  # -5,1
    plt.show()
