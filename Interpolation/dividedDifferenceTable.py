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

def getDDTableDiagonal(points):
    rows = 3 * len(points)
    #   create a divided differences table
    ddTable = [[points[trunc(row/3)][0],
                points[trunc(row/3)][1]]
               for row in range(0,rows)]

    #   populate first derivatives
    for row in range(0,rows - 1):
        diff = ddTable[row+1][1] - ddTable[row][1]
        if diff == 0:
            ddTable[row].append(points[trunc(row/3)][2])
        else:
            diff /= (ddTable[row+1][0] - ddTable[row][0])
            ddTable[row].append(diff)
    ddTable[-1].append(0)

    #   populate second derivatives
    for row in range(0,rows - 2):
        diff = ddTable[row+1][2] - ddTable[row][2]
        if diff == 0:
            ddTable[row].append(points[trunc(row/3)][3] / 2)
        else:
            diff /= (ddTable[row+2][0] - ddTable[row][0])
            ddTable[row].append(diff)
    ddTable[-2].append(0)
    ddTable[-1].append(0)

    #   populate remainder of table
    for col in range(4, rows + 1):
        for row in range(0,rows):
            if row > rows - col:
                ddTable[row].append(0)
            else:
                diff = ddTable[row+1][col-1] - ddTable[row][col-1]
                diff /= (ddTable[(row+col)-1][0] - ddTable[row][0])
                ddTable[row].append(diff)
    #   return the 'diagonal' of the table
    return(ddTable)

ddTable = getDDTableDiagonal(points)
ddRows = ddTable[0][1:]

print("Execution Complete")