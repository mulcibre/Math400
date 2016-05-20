import numpy
from math import e
import matplotlib.pyplot as plt
import scipy.optimize as optimization

xdata = numpy.array([0.0100, 0.9998, 2.1203, 3.0023, 3.9892, 5.0017])
ydata = numpy.array([1.9262, 1.0442, 0.4660, 0.2496, 0.0214, 0.0130])
x0    = numpy.array([0.0, 0.0])
def f(x, a, b):
    return b*e**(-a*x)

# using library function from numpy
sol, _ = optimization.curve_fit(f, xdata, ydata, x0)
a, b = sol

x = numpy.linspace( -1, 6, num=500 )
vf = numpy.vectorize(f)

# definitions of error and
def err(a,b,x,y):
  return (y - f(x,a,b))**2
def error(f,a,b):
  sum = 0;
  for x, y in zip(xdata, ydata):
    sum += err(a,b,x,y)
  return sum
print error(f,a,b)

plt.plot( xdata, ydata, 'o')
plt.plot( x, vf(x, a, b), '-')
plt.legend(['data', 'fitted(numpy)'],loc='best')
plt.axhline(y=0, color='black')
plt.savefig('leastsquares')
plt.clf()