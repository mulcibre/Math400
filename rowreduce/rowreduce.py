#   uncomment the line below to enable debug mode
#import pdb; pdb.set_trace()
import math

def tellUserHowToPivotStrategy():
    print("Please enter valid pivot strategy")
    print("0 -- Naive Gaussian")
    print("1 -- Partial Pivoting")
    print("2 -- Scaled Pivoting")

def GetRow(mat, row):
    #return row from matrix mat
    retVec=[mat[row][j] for j in range(len(mat[0]))]
    return(retVec)

def GetCol(mat, col):
    "return column col from matrix mat"
    retVec=[mat[j][col] for j in range(len(mat))]
    return(retVec)
    
def SubtractMatrices(A,B):
    "add two matrices, checking first that correct dimensions match"
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        print("matrix dimensions must match for matrix addition")
    else:
        retMat = [[A[row][col] - B[row][col] for col in range(len(A[0]))] for row in range(len(A))]    
        return(retMat)

def Dot(S,T):
    "return dot product of two vectors"
    if len(S) != len(T):
        print("for dot product, vector dimensions must match")
    else:
        retVal = 0
        for j in range(len(S)):
            retVal += S[j] * T[j]
        return(retVal)
        
def MultiplyMat(mat1,mat2):
    "multiply two matrices, checking first that correct dimensions match"
    if len(mat1[0]) != len(mat2):
        print("row count of first matrix must match col count of second matrix for multiplication")
    else:
        retMat = [[Dot(GetRow(mat1, row),GetCol(mat2, col)) for col in range(len(mat2[0]))] for row in range(len(mat1))]    
        return(retMat)    
    
def swapRows(mat, row1, row2):
    #   store second row values first
    temp = GetRow(mat, row2)
    #   then swap
    mat[row2] = GetRow(mat, row1)
    mat[row1] = temp
    
def getPivotValue(mat, rowIndex, colIndex, pivotStrategy):
    #   return pivot value, or scaled pivot value depending on strategy
    numerator = abs(mat[rowIndex][colIndex])
    if pivotStrategy == 1:
        #   return just the pivot column element for partial pivoting
        return(numerator)
    if pivotStrategy == 2:
        #   return the pivot col value, normalized by all the elements in the row
        #   (except of course the final solution element)
        denominator = sum(abs(i) for i in mat[rowIndex]) - abs(mat[rowIndex][-1])
        return(numerator / denominator)
    else:
        print("invalid pivoting strategy")

def getBestPivot(mat, pivotCol, pivotStrategy):
    #   set initial values for the pivot row, and pivot value
    maxPivotValue = getPivotValue(mat, pivotCol, pivotCol, pivotStrategy)
    maxPivotRow = pivotCol
    #   Check the rest of the rows for a higher pivot value
    for i in range(pivotCol + 1, len(mat)):
        #   get scaled, or partial pivot value, depending on strategy
        newPivotValue = getPivotValue(mat, i, pivotCol, pivotStrategy)
        if newPivotValue > maxPivotValue:
            #   set new max
            maxPivotValue = newPivotValue
            maxPivotRow = i
    #   now the new max pivot row can be swapped in, if there is one
    if maxPivotRow != pivotCol:
        swapRows(mat, pivotCol, maxPivotRow)
    
def setBestPivotRow(mat, pivotColIndex, pivotStrategy):
    if pivotStrategy == 0:
        #   no strategy
        return
    else:
        #   partial or scaled pivoting
        getBestPivot(mat, pivotColIndex, pivotStrategy)
        return
    
def showMatrix(mat):
    "Print out matrix"
    for row in mat:
        print(row)

#   This function uses a pivot row to remove that pivot value
#   From another row
def removePivotValueFromRow(col, pivotRow, rowToClean):
    pivotRatio = rowToClean[col]
    if pivotRatio != 0:
        for i in range(len(pivotRow)):
            rowToClean[i] -= pivotRow[i] * pivotRatio
    return(rowToClean)
    
#   This function will divide every value in the row by the pivot value
#   The result is that the pivot is changed to 1
def setPivotToOne(col, pivotRow):
    if pivotRow[col] != 0:
        divisor = pivotRow[col]
        for i in range(col, len(pivotRow)):
            pivotRow[i] /= divisor
    return(pivotRow)
       
def solveMatrix(inMat, pivotStrategy):
    mat = [[inMat[row][col] for col in range(len(inMat[0]))] for row in range(len(inMat))]
    if pivotStrategy != 0 and pivotStrategy != 1 and pivotStrategy != 2:
        tellUserHowToPivotStrategy()
        sys.exit(1)

    #    for each row in the matrix
    for i in range(len(mat)):
        #   0: no pivot strategy
        #   1: partial pivot strategy
        #   2: scaled pivot strategy
        setBestPivotRow(mat, i, pivotStrategy)
        #    set pivot in row to 1
        setPivotToOne(i, mat[i])
        #    remove value in pivot column from other rows    
        for j in range(len(mat)):
            if j != i:
                removePivotValueFromRow(i, mat[i], mat[j])
    return(mat)

def getErrorVec(originalMat, solutionMat):

    #   get final column of original matrix, b
    b = [originalMat[row][len(originalMat[0])-1] for row in range(len(originalMat))]
    
    #   get square component of original matrix, A
    A = [row[0:len(originalMat[0])-1] for row in originalMat]
    
    #   get xTilde, the solution vector
    xTilde = GetCol(solutionMat, len(solutionMat[0]) - 1)
    
    #   compute r
    #   multiply step
    retVec = [Dot(A[row],xTilde) for row in range(len(A))]
    #   subtraction step
    retVec = [b[i] - retVec[i] for i in range(len(b))]
    return(retVec)

######  Initial tests

A= [[4,-2,1,11],
    [-2,4,-2,-16],
    [1,-2,4,17]]
    
testMat =  [[1.,1.,1.,6.],
            [0.,2.,5.,-4.],
            [2.,5.,-1.,27.]]
            
testMatDesiredResult = [[1,0,0,5],
                        [0,1,0,3],
                        [0,0,1,-2]]
       
#   Source augmented matrix for homework part 2       
part2Mat =  [[1.000,1.000,1.000,0.001,3.0],
             [1.000,0.001,0.001,0.001,1.0],
             [1.000,1.000,0.001,0.001,2.0],
             [10.00,10.00,10.00,math.pow(10,17),math.pow(10,17)]]
                
def testRowOps():
    print("\nMatrix")
    showMatrix(testMat)
    print("\nAfter row 3 has pivot1 removed")
    testMat[2] = removePivotValueFromRow(0, testMat[0], testMat[2])
    showMatrix(testMat)
    
def testMatrix():
    print("Matrix")
    showMatrix(part2Mat)
    print
    print("solved matrix with naive Gaussian elimination:")
    solved = solveMatrix(part2Mat, 0)
    showMatrix(solved)
    errorVec = getErrorVec(part2Mat, solved)
    print("error vector r for naive Gaussian elimination")
    showMatrix(errorVec)
    print
    print("solved matrix with partial pivoting:")
    solved = solveMatrix(part2Mat, 1)
    showMatrix(solved)
    errorVec = getErrorVec(part2Mat, solved)
    print("error vector r for partial pivoting")
    showMatrix(errorVec)
    print
    print("solved matrix with scaled pivoting:")
    solved = solveMatrix(part2Mat, 2)
    showMatrix(solved)
    errorVec = getErrorVec(part2Mat, solved)
    print("error vector r for scaled pivoting")
    showMatrix(errorVec)
    
def testPivotOps():
    print("\nMatrix for part 2:")
    showMatrix(part2Mat)
    print("\nRow 1 and 3 swapped:")
    swapRows(part2Mat,0,2)
    showMatrix(part2Mat)
    print("\nGet best row for pivot 1 by partial pivoting:")
    setBestPivotRow(part2Mat, 0, 0)
    showMatrix(part2Mat)
    
#testRowOps()
testMatrix()
#testPivotOps()