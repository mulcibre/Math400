#   Math 400
#   Assignment 4
#   Problem 2
#   Samuel Gluss
#   Algorithm Credit: Emily Conway
#   4-25-2016

import math

###
#   Eulers for second order ODE
###

def pT(table):
    for i in range(0,len(table)):
        print ("%.2f" % table[i][0]) + "," + str(table[i][1])

def printVals(table):
    print ",".join(map(str, rk4[:][1]))

def f1(w2):
    #   in this case w1 is not used, but for semantic accuracy we take both w vals as params
    retVal = w2
    return retVal

def f2(w1):
    retVal = (-4.0 * (math.pi ** 2) * w1)
    return retVal

#   generate tk values
h = 0.01
upper = int(round(1.2 / h)) + 1
eulers = [[x * h,0,0] for x in range(0, upper, 1)]

#   establish initial values for xk and yk
eulers[0][1] = -1.0
eulers[0][2] = 0.0

#   populate table
for i in range(1, len(eulers)):
    eulers[i][1] = eulers[i-1][1] + h * f2(eulers[i-1][2])
    eulers[i][2] = eulers[i-1][2] + h * f1(eulers[i-1][1])

print "Eulers results:\n"
pT(eulers)

###
#   RK4 for second order ODE
###

#   generate tk values
h = 0.1
upper = int(round(1.2 / h)) + 1

#   table format: t, x, k11, k12, k13, k14, y, k21, k22, k23, k24
rk4 = [[x * h,0,0,0,0,0,0,0,0,0,0] for x in range(0, upper, 1)]

#   please see Burden + Faires 9th edition page 335
#   establish initial values for w1 and w2
w1 = rk4[0][1] = -1
w2 = rk4[0][2] = 0

#   k1,1
rk4[0][3] = h * f1(w2)
#   k1,2
rk4[0][4] = h * f2(w1)
#   k2,1 = h * f1(w2 + 0.5 * k1,2)
rk4[0][5] = h * f1(w2 + 0.5 * rk4[0][4])
#   k2,2 = h * f2(w1 + 0.5 * k1,1)
rk4[0][6] = h * f2(w1 + 0.5 * rk4[0][3])
#   k3,1 = h * f1(w2 + 0.5 * k2,2)
rk4[0][7] = h * f1(w2 + 0.5 * rk4[0][6])
#   k3,2 = h * f2(w1 + 0.5 * k2,1)
rk4[0][8] = h * f2(w1 + 0.5 * rk4[0][5])
#   k4,1 = h * f1(w2 + 0.5 * k3,2)
rk4[0][9] = h * f1(w2 + 0.5 * rk4[0][8])
#   k4,2 = h * f2(w1 + 0.5 * k3,1)
rk4[0][10] = h * f2(w1 + 0.5 * rk4[0][7])

#   populate table (see lecture notes page 103 and 107)
for i in range(1, len(rk4)):
    #   generate next w1, w2 vals
    #   w1,i = w1,i-1 + (1/6)(k1,1 + 2k2,1 + 2k3,1 + k4,1)
    w1 = rk4[i][1] = rk4[i - 1][1] + ((rk4[i - 1][3] + (2 * rk4[i - 1][5]) + (2 * rk4[i - 1][7]) + rk4[i - 1][9]) / 6)
    #   w2,i = w2,i-1 + (1/6)(k1,2 + 2k2,2 + 2k3,2 + k4,2)
    w2 = rk4[i][2] = rk4[i - 1][2] + ((rk4[i - 1][4] + (2 * rk4[i - 1][6]) + (2 * rk4[i - 1][8]) + rk4[i - 1][10]) / 6)

    #   k1,1
    rk4[i][3] = h * f1(w2)
    #   k1,2
    rk4[i][4] = h * f2(w1)
    #   k2,1 = h * f1(w2 + 0.5 * k1,2)
    rk4[i][5] = h * f1(w2 + 0.5 * rk4[i][4])
    #   k2,2 = h * f2(w1 + 0.5 * k1,1)
    rk4[i][6] = h * f2(w1 + 0.5 * rk4[i][3])
    #   k3,1 = h * f1(w2 + 0.5 * k2,2)
    rk4[i][7] = h * f1(w2 + 0.5 * rk4[i][6])
    #   k3,2 = h * f2(w1 + 0.5 * k2,1)
    rk4[i][8] = h * f2(w1 + 0.5 * rk4[i][5])
    #   k4,1 = h * f1(w2 + 0.5 * k3,2)
    rk4[i][9] = h * f1(w2 + 0.5 * rk4[i][8])
    #   k4,2 = h * f2(w1 + 0.5 * k3,1)
    rk4[i][10] = h * f2(w1 + 0.5 * rk4[i][7])

###
#   Table generation complete
###

#   print table
print "\n\nRK4 results:\n"
pT(rk4)