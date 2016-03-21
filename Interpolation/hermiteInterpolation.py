#   Samuel Gluss                        #
#   3/12/2016                           #
#   Math 400 Assignment 2 part 2        #
#   Generalized Hermite Interpolation   #
#########################################
import sys
from math import trunc

#   each point has xi, fxi, f1xi, f2xi
#   some points for f(x) = x^8 + 1
points = [[-1.,2.,-8.,56.],
          [0.,1.,0.,0.],
          [1.,2.,8.,56.]]

testPoly1 = [[1,4],[5,3],[2,1],[15,0]]
testPoly2 = [[3,7],[2,3],[1,2],[12,0]]

def printPoly(poly):
    for i in range(0,len(poly)):
        #   write the coefficient
        sys.stdout.write(str(poly[i][0]))
        #   write an x if appropriate
        if poly[i][1] > 0:
            sys.stdout.write('x')
        #   write power if appropriate
        if poly[i][1] > 1:
            sys.stdout.write('^' + str(poly[i][1]))
        #   if not last element, output a plus
        if i < len(poly) - 1:
            sys.stdout.write(' + ')
    print

#   modifies poly1 to add poly2
def addPoly(poly1, poly2):
    i = 0
    j = 0
    poly3 = []
    if not poly1:
        return poly2
    while i < len(poly1) or j < len(poly2):
        #   cases that one polynomial is completely consumed first
        if i == len(poly1):
            while j < len(poly2):
                poly3.append(poly2[j])
                j+=1
        elif j == len(poly2):
            while i < len(poly1):
                poly3.append(poly1[i])
                i += 1
        #   if two elements at same power are found, add coefficients
        elif poly1[i][1] == poly2[j][1]:
            poly3.append([poly1[i][0] + poly2[j][0],poly1[i][1]])
            i += 1
            j += 1
        #   otherwise choose one
        elif poly1[i][1] > poly2[j][1]:
            poly3.append(poly1[i])
            i += 1
        else:
            poly3.append(poly2[j])
            j += 1
    return poly3

def multPoly(poly1, poly2):
    poly3 = []
    for i in range(0,len(poly1)):
        for j in range(0,len(poly2)):
            product = [[poly1[i][0] * poly2[j][0], poly1[i][1] + poly2[j][1]]]
            poly3 = addPoly(poly3, product)
    return poly3

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

print "testing addition"
printPoly(testPoly1)
printPoly(testPoly2)
testPoly1 = addPoly(testPoly1,testPoly2)
printPoly(testPoly1)

print "\ntesting multiplication"
printPoly(multPoly([[1,2],[-3,0]],[[3,3],[1,1]]))
print("Execution Complete")