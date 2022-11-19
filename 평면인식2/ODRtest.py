import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import random
import scipy.odr

def first():
    # Initiate some data, giving some randomness using random.random().
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([i**2 + random.random() for i in x])

    # Define a function (quadratic in our case) to fit the data with.
    def linear_func(p, x):
        m, c = p
        return m*x + c

    # Create a model for fitting.
    linear_model = Model(linear_func)

    # Create a RealData object using our initiated data from above.
    data = RealData(x, y)

    # Set up ODR with the model and data.
    odr = ODR(data, linear_model, beta0=[0., 1.])

    # Run the regression.
    out = odr.run()

    # Use the in-built pprint method to give us results.
    out.pprint()


    from scipy import odr
    x = np.linspace(0.0, 5.0)
    y = 10.0 + 5.0 * x + random.random()
    data = odr.Data(x, y)
    odr_obj = odr.ODR(data, odr.multilinear)
    output = odr_obj.run()
    print(output.beta)




def second():
    n = 1000
    t = np.linspace(0, 1, n)

    def linfit(beta, x):
        return beta[0]*x[0] + beta[1]*x[1] + beta[2] #notice changed indices for x

    x1 = 2.5*np.sin(2*np.pi*6*t)+4
    x2 = 0.5*np.sin(2*np.pi*7*t + np.pi/3)+2

    x = np.row_stack( (x1, x2) ) #odr doesn't seem to work with column_stack

    e = 0.25*np.random.randn(n)
    y = 3*x[0] + 4*x[1] + 5 + e #indices changed

    linmod = scipy.odr.Model(linfit)
    data = scipy.odr.Data(x, y)
    odrfit = scipy.odr.ODR(data, linmod, beta0=[1., 1., 1.])
    odrres = odrfit.run()
    #odrres.pprint()
    print(odrres.beta)

    fig = plt.figure(0, figsize=(4, 3))
    #plt.clf()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0, 0, 0.95, 1])

    ax.scatter(x1[::10], x2[::10], y[::10])
    plt.show()


def ODRline():
    
    # X = np.array([p.x for p in pts])
    # Y = np.array([p.y for p in pts])
    # Z = np.array([p.z for p in pts])
    initial = np.array([1,2,3])
    direction = np.array([2,1,-1])
    t = np.linspace(0, 3, 1000)
    X = np.array([initial[0] + direction[0]*step for step in t])
    Y = np.array([initial[1] + direction[1]*step for step in t])
    Z = np.array([initial[2] + direction[2]*step for step in t])
    y = np.c_[X,Y,Z]

    def f(B, x):
        return B[0]*x + B[1]
    
    linmod = scipy.odr.Model(f)
    data = scipy.odr.Data(t, Y)
    odrfit = scipy.odr.ODR(data, linmod, beta0=[np.array([1., 1., 1.]), np.array([1.,1.,1.])])
    odrres = odrfit.run()
    beta = odrres.beta
    odrres.pprint()
    return (beta[0], beta[1], -1, beta[2])

def ODRplane(pts, dummy = None):
    X = np.array([p.x for p in pts])
    Y = np.array([p.y for p in pts])
    Z = np.array([p.z for p in pts])

    def linfit(beta, x): #x = (X,Y)
        return beta[0]*x[0] + beta[1]*x[1] + beta[2]
    x = np.row_stack( (X, Y) )

    linmod = scipy.odr.Model(linfit)
    data = scipy.odr.Data(x, Z)
    odrfit = scipy.odr.ODR(data, linmod, beta0=[1., 1., 1.])
    odrres = odrfit.run()
    beta = odrres.beta
    return (beta[0], beta[1], -1, beta[2])

ODRline()