#   Samuel Gluss                        #
#   3/12/2016                           #
#   Math 400 Assignment 2 part 2        #
#   Generalized Hermite Interpolation   #
#########################################

from math import trunc

#   each point has xi, fxi, f1xi, f2xi
#   some points for f(x) = x^8 + 1
points = [[-1.,2.,-8.,56.],
          [0.,1.,0.,0.],
          [1.,2.,8.,56.]]

    #               (x - x0) . . .(x - xk-1)(x - xk+1) . . .(x - xn)
    #   Ln,k(x) =   --------------------------------------------------
    #               (xk - x0) . . .(xk - xk-1)(xk - xk+1) . . .(xk - xn)
def nthLagrange(n,k,x,xlist):
    numerator = 1
    denominator = 1
    for i in range(0,len(xlist)):
        if(i != k):
            numerator *= (x - xlist[i])
            denominator *= (xlist[k] - xlist[i])
    return numerator / denominator

print("Execution Complete")