"""
Author:      Jose Ortiz Costa
Class:       Math400 Assigment: 2
Group:       07
Date:        03/12/2016
Description: This script provide a numerical analysis for several 
             interpolation methods using a set of x data for the x's axis, 
             and the function g(x) = 2*e^-2x*sin(3pix) evaluated for this x's 
             points. This script also allow to take the difference between 
             the function g(x) and the choose interpolated method when 
             plotting them.
             The interpolation methods aviable in this script are:
             1. Larrange Interpolation.
             2. Hermite Interpolation
             3. Piecewise Linear Interpolation
             4. Cubic Spline Interpolation
             
             This script also prints in console x, g(x), g'(x) and the 
             polynomial result functions from all the methods tested here.
             
             Note: This script is not finished because it still need to 
             print in console the polynomial interpolation function  of
             hermite as well as the functions from piecewise linear method
             and cubic spline method.
"""
#-------------------------- Functions -----------------------------------------
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import derivative
import numpy as np
import math
from math import *

# Constants
DIFF_CONSTANT = 50
INTERPOLANT_POINTS_NUMBER = 1000
ARRANGE_SPACES = 0.001
INTERVAL_X_BEGINS_AT = 0.0
INTERVAL_X_ENDS_AT = 1.0

# Enumerator class that defines every interpolant method
class InterpolantMethod:
    LAGRANGE = 0
    HERMITE = 1
    PIECEWISE_LINEAR = 2
    CUBIC_SPLINE =3

# Computes gx for a given list of x values
def g(x):
  "Returns an array of g(x) values for the g function 2e^-2x * sin(3*pi*x)"
  return [(2*(math.pow(math.e, -2*value)) * math.sin(3*math.pi*value)) for value in x]

# Returns the Larrange Interpolation Polynomial 
def lagrangeInterpolation (x, gx):
   #setting result = 0
   result = scipy.poly1d([0.0]) 
   for i in range(0,len(x)): #number of polynomials L_k(x).
      temp_numerator = scipy.poly1d([1.0]) # resets temp_numerator such that a new numerator can be created for each i.
      denumerator = 1.0 #resets denumerator such that a new denumerator can be created for each i.
      for j in range(0,len(x)):
          if i != j:
              temp_numerator *= scipy.poly1d([1.0,-x[j]]) #finds numerator for L_i
              denumerator *= x[i]-x[j] #finds denumerator for L_i
      result += (temp_numerator/denumerator) * gx[i] #linear combination
   return result;

# find the coordinate where the difference is greater
def findMaxDiff (difference):
    return max(difference)/DIFF_CONSTANT, max(difference)*ARRANGE_SPACES

#Calculates g'(x)  = first derivative of g(x)
def function(x):
     return (2*(e**(-2*x)) * sin(3 *  pi * x))

# find lagrange function derivate
def lagrange_derivative(k, xNodes):
    sum = 0
    for j in range(len(xNodes)):
        if j != k:
            sum = sum + 1.0 * 1 / (xNodes[k] - xNodes[j])
    return sum

# hermite interpolation function
def hermiteInterpolation(x, nodes, values, deriv):
    hermite = 0
    for k in range(len(nodes)):
        num = 1;
        denom = 1;
        for j in range(len(nodes)):
            if nodes[j] == nodes[k]:
               continue
            num = num * (x - nodes[j])
            denom = denom * (nodes[k] - nodes[j])
        derivative = lagrange_derivative(k, nodes)
        # hermite left part of H(x)
        hermite_l =  ((1.0 * (num / denom))*(1.0 * (num / denom)))*(1 - (2 * derivative * (x - nodes[k])))
        # hermite right part of H(X)       
        hermite_r = ((1.0 * (num / denom))*(1.0 * (num / denom))) * (x - nodes[k])
        hermite += (hermite_l * values[k]) + (hermite_r * deriv[k]) 
    return hermite
           

# Returns the Piecewise Linear Interpolation Polynomial
def piecewiseLinearInterpolation(x, gx):
    return interp1d(x, gx, kind='linear')

# Returns the Cubic Spline Interpolation Polynomial
def cubicSplineInterpolation(x,gx):
    return interp1d(x, gx, kind='cubic')


# Returns a function g(x) ready to be derivated in funcDerivate() method
def func (x):
    return 2*math.pow(math.e, -2*x) * math.sin(3*math.pi*x) 

# Find the first derivate of a function defined in func(x) method 
def funcDerivate(xValues):
    return [derivative(func, xv, dx=1e-6) for xv in xValues]   
    
    
# Creates a plot with curves g(x), interpolation method chosen and optionally
# computer the difference between them
def plot (x, gx, method=InterpolantMethod.LAGRANGE, computeDiff=False, plotTitle=None, 
          labelx=None, labely=None, legendInterpolation=None, legendFX=None, 
          legendDiff=None):
    xvalues = np.arange(min(x),max(x), ARRANGE_SPACES) # arrange x values
    realCurveX = np.linspace(INTERVAL_X_BEGINS_AT,INTERVAL_X_ENDS_AT,INTERPOLANT_POINTS_NUMBER) # 1000 spaces numbers between 0 and 1.0
    
     # Checks which method was chosen to be computed in this function
    if method == InterpolantMethod.LAGRANGE:
        interpolation = lagrangeInterpolation(x, gx)
        interpolation = interpolation(xvalues)
    elif method == InterpolantMethod.HERMITE:
        interpolation =  [hermiteInterpolation(interpolant, x, gx, funcDerivate(x)) for interpolant in realCurveX]
    elif method == InterpolantMethod.PIECEWISE_LINEAR:
        interpolation = piecewiseLinearInterpolation(x, gx)
        interpolation = interpolation(xvalues)
    elif method == InterpolantMethod.CUBIC_SPLINE:
        interpolation = cubicSplineInterpolation(x,gx)
        interpolation = interpolation(xvalues)
    else:
        print "Error: Interpolation methods available are: "
        print "1. Larrange\n2. Hermite\n3. Piecewise_Linear\n4. Cubic_Spline"
        return
    # if true, computes difference between the curves g(x) 
    # and the interpolation method chosen
    if computeDiff is True:
         difference = np.absolute(np.subtract(g(xvalues), interpolation))
         difference = [DIFF_CONSTANT*i for i in difference]
         print (" Maximum difference found at coordinate: ", 
               findMaxDiff(difference))
    # setting titles and labels
    plt.title(plotTitle)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    #plotting 
    plt.plot(x, gx, 'o', realCurveX, 
             g(realCurveX), '-',xvalues, interpolation, '--',  xvalues, difference, '-')
    plt.legend(['data', legendFX, legendInterpolation, legendDiff], loc='best')
    plt.show()
    
    
    
#------------------ Test Script -----------------------------------------------

#array of xvalues given [x0, x1, x2, ... , x10]
x = scipy.array([i/24. for i in [0, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]])

# prints x, g(x), and g'(x)

print "Values of x: \n"
print x 
print "\nValues of g(x): \n"
print g(x) 
print "\nValues of g'(x): \n"
print funcDerivate(x)

# Plot figures containing the four interpolation methods

# Plot larrange interpolation figure.
plt.figure(1)
plot(x,g(x), InterpolantMethod.LAGRANGE, True, "g(x) vs. Lagrange Interpolation", "x", "g(x)", 
             "Lagrange curve", "g(x) Function curve", "Difference Curve" )

# Plot hermite interpolation figure   
plt.figure(2)
plot(x,g(x), InterpolantMethod.HERMITE, True, "g(x) vs. Hermite Interpolation Plot", "x", "g(x)", 
             "Hermite curve", "g(x) Function curve", "Difference Curve" )

# Plot Piecewise Linear interpolation figure
plt.figure(3)
plot(x,g(x), InterpolantMethod.PIECEWISE_LINEAR, True, "g(x) vs. Piecewise Linear Interpolation Plot", 
             "x", "g(x)", "Piecewise Linear Interpolation Curve", 
             "g(x) Function curve", "Difference Curve" )

# Plot Cubic Spline interpolation figure.
plt.figure(4)
plot(x,g(x), InterpolantMethod.CUBIC_SPLINE, True, "g(x) vs. Cubic Spline Interpolation Plot", "x", 
             "g(x)", "Cubic Spline Interpolation Curve", "g(x) Function curve", 
             "Difference Curve" )
             
