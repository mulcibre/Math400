#   uncomment the line below to enable debug mode
#import pdb; pdb.set_trace()

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
    
def solveMatrix(mat):
    #    for each row in the matrix
    for i in range(len(mat)):
        #    set pivot in row to 1
        setPivotToOne(i, mat[i])
        #    remove value in pivot column from other rows    
        for j in range(len(mat)):
            if j != i:
                removePivotValueFromRow(i, mat[i], mat[j])
    return(mat)

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
    print("Matrix")
    showMatrix(testMat)
    print("After row 3 has pivot1 removed")
    testMat[2] = removePivotValueFromRow(0, testMat[0], testMat[2])
    showMatrix(testMat)
    
def testMatrix():
    print("Matrix")
    showMatrix(testMat)
    print("solved matrix")
    showMatrix(solveMatrix(testMat))
    
#testRowOps()
testMatrix()