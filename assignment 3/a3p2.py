#	Samuel Gluss
#	4-3-2016
#	Assignment 3
#	number 2
#import pdb; pdb.set_trace()
import math

points =    [[1.0, 1.00000000],
            [1.1, 0.90483742],
            [1.2, 0.81873075],
            [1.3, 0.74081822],
            [1.4, 0.67032005],
            [1.5, 0.60653066],
            [1.6, 0.54881164],
            [1.7, 0.49658530],
            [1.8, 0.44932896]]

#test result should be 1
points1 =    [[1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [1.3, 1.3],
            [1.4, 1.4],
            [1.5, 1.5],
            [1.6, 1.6],
            [1.7, 1.7],
            [1.8, 1.8]]

#   average of two rectangles is the trapezoid volume            
def subIntVol(point1, point2):
    width = point2[0]-point1[0]
    rect1Vol = width*point1[1]
    rect2Vol = width*point2[1]
    return float(rect1Vol + rect2Vol) / 2

    #   trapezoid integration result is the sum of all subintervals
def trapezoid(points):
    retVal = 0
    for i in range(0, len(points) - 1):
        retVal += subIntVol(points[i],points[i+1])
    return retVal
    
def CDF(points, xIndex, hOffset):
    rightFx = points[xIndex + hOffset][1]
    leftFx = points[xIndex - hOffset][1]
    retVal = rightFx - leftFx
    h = points[xIndex + hOffset][0] - points[xIndex][0]
    retVal /= (2 * h)
    return retVal

def pT(table):
    for row in table:
        print row

def Li(points, iIndex, xIndex):
    x = points[xIndex][0]
    xi = points[iIndex][0]
    retVal = 0

    numer = 1
    denom = 1
    #   compute coefficient for yi
    for i in range(0, len(points)):
        if i != iIndex and i != xIndex:
            numer *= x - points[i][0]
            denom *= xi - points[i][0]
    #   multiply coeff by yi, add to return value
    retVal = numer / denom
    return retVal

def LiPrime(points, iIndex, xIndex):
    retVal = 0
    x = points[xIndex][0]

    for m in range(0, len(points)):
        if m != iIndex and m != xIndex:
            retVal += 1 / (x - points[m][0])
    retVal *= Li(points, iIndex, xIndex)
    return retVal

#   second formula for Lprime on math stackexchange
def LiPrime2(points, iIndex, xIndex):
    x = points[xIndex][0]
    xi = points[iIndex][0]
    outersum = 0.0
    for i in range(0, len(points)):
        numerProduct = 1.0
        for j in range(0, len(points)):
            if j != i and j != xIndex:
                next = (x - points[j][0])
                numerProduct *= next
        outersum += numerProduct

    denomProduct = 1
    for k in range(0, len(points)):
        if k != iIndex:
            denomProduct *= (xi - points[k][0])
    return outersum / denomProduct

def getSlope(points, xIndex):
    yPrime = 0
    for i in range(0, len(points)):
        yPrime += points[i][1] * LiPrime2(points, i, xIndex)
    return yPrime

def midPointFormula(points):
    if not len(points)%2:
        print("number of points must be odd")
        exit(-1)
    else:
        midIndex = (len(points) - 1) / 2
        yPrime = 0

        for i in range(0, len(points)):
            toAdd = points[i][1]

            if i < midIndex:
                toAdd *= LiPrime(points, i, midIndex)
                yPrime -= toAdd
            elif i > midIndex:
                toAdd *= LiPrime(points, i, midIndex)
                yPrime += toAdd
        yPrime /= (math.factorial(len(points) - 1) * (points[1][0] - points[0][0]))
        return yPrime

#   found equation for derivate at construction point here:
#   http://math.stackexchange.com/questions/1105160/evaluate-derivative-of-lagrange-polynomials-at-construction-points    
#   find derivate at a point according to:
#                   n
#   Li1(x) = Li(x) * E  1/(x - xm)
#                   m=0, m!=i
def lagrangePrime(points, xIndex):
    Lix = Li(points, xIndex)
    summation = 0
    for i in range(0, len(points)):
        if i != xIndex:
            summation += 1 / (points[xIndex][0] - points[i][0])
    return Lix * summation

#trapezoidal integral
#print trapezoid(points)

hOffset = 4
xIndex = 4
table = [[0 for x in range(0,3)] for y in range(0,3)]

for n in range(0,3):
    #   set leftmost value in row
    table[n][0] = CDF(points, xIndex, hOffset / 2**n)
    for j in range(1, n + 1):
        #   grab necessary table values to compute Dj(hn)
        leftOne = table[n][j-1]
        leftOneUpOne = table[n-1][j-1]
        table[n][j] = leftOne + ((leftOne - leftOneUpOne)/(2**(2*(j)) - 1))

#   print table of values 
print "Most accurate estimation is on the bottom right:"         
pT(table)    
    
print "\nThe error is: " + str(abs(table[-1][-1] - table[-1][-2]))

# test result should be 1
points = [[1.0, 1.0],
          [2.0, 3.0],
          [3.0, 5.0]]

print "\nBy Lagrange Interpolation: " + str(midPointFormula(points))
    