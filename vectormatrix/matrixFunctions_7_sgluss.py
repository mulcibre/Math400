"""
matrixFunctions.py

2/1/16 bic M400

Assignment completed 2/4/2016 by Samuel Gluss
"""

#######  START Administrivia 
m400group = 7   # change this to your group number

m400names = ['*Samuel Gluss', 'Emily Conway', 'Jose Ortiz Costa', 'Abdelmajid Samir', 'Edward Yao'] # change this for your names

def printNames():
    print("matrixFunctions.py for group %s:"%(m400group)),
    for name in m400names:
        print("%s, "%(name)),
    print

printNames()

#######  END Administrivia


"""
Vector Functions

copy these three functions from your finished vectorFunctions.py file

"""

def ScalarMult(s,V):
    "return vector sV"
    sV = [(s * V[j]) for j in range(len(V))]
    return(sV)


def AddVectors(S,T):
    "return S+T"
    if len(S) == len(T):
        retVec = [(S[j] + T[j]) for j in range(len(S))]
        return(retVec)
    else:
        print("for adding, vector dimensions must match")


def Dot(S,T):
    "return dot product of two vectors"
    if len(S) != len(T):
        print("for dot product, vector dimensions must match")
    else:
        retVal = 0
        for j in range(len(S)):
            retVal += S[j] * T[j]
        return(retVal)


"""

Matrix Functions

"""



def showMatrix(mat):
    "Print out matrix"
    for row in mat:
        print(row)


def rows(mat):
    "return number of rows"
    return(len(mat))

def cols(mat):
    "return number of cols"
    return(len(mat[0]))
 

#### Functions for you to finish

def GetCol(mat, col):
    "return column col from matrix mat"
    retVec=[mat[j][col] for j in range(rows(mat))]
    return(retVec)

def Transpose(mat):
    "return transpose of mat"
    retMat = [[mat[row][col] for row in range(rows(mat))] for col in range(cols(mat))]    
    return(retMat)

def GetRow(mat, row):
    "return row row from matrix mat"
    retVec=[mat[row][j] for j in range(cols(mat))]
    return(retVec)

def ScalarMultMatrix(a,mat):
    retMat = [[mat[row][col] * a for col in range(cols(mat))] for row in range(rows(mat))]    
    return(retMat)


def AddMatrices(A,B):
    "add two matrices, checking first that correct dimensions match"
    if rows(A) != rows(B) or cols(A) != cols(B):
        print("matrix dimensions must match for matrix addition")
    else:
        retMat = [[A[row][col] + B[row][col] for col in range(cols(A))] for row in range(rows(A))]    
        return(retMat)

        
def MultiplyMat(mat1,mat2):
    "multiply two matrices, checking first that correct dimensions match"
    if cols(mat1) != rows(mat2):
        print("row count of first matrix must match col count of second matrix for multiplication")
    else:
        retMat = [[Dot(GetRow(mat1, row),GetCol(mat2, col)) for col in range(cols(mat2))] for row in range(rows(mat1))]    
        return(retMat)


######  Initial tests

A= [[4,-2,1,11],
    [-2,4,-2,-16],
    [1,-2,4,17]]

Ae= [[4,-2,1],
    [-2,4,-2],
    [1,-2,4]]


Bv=[11,-16,17]

Bm=[[11,-16,17]]

C=[2,3,5]

print("running matrixFunction.py file")

#   Some additional test functions to speed up debugging
def testMatrixFuncs():
    print("matrix A:")
    showMatrix(A)
    print("first col of A:")
    print("%s"%GetCol(A,0))
    print("first row of A:")
    print("%s"%GetRow(A,0))
    print("Transpose of A:")
    showMatrix(Transpose(A))

def testMatrix():
    print("A")
    showMatrix(A)
    print("Bm")
    showMatrix(Bm)
    print("Ae")
    showMatrix(Ae)
    print("multiplyMat(Ae,A)")
    showMatrix(MultiplyMat(Ae,A))
    print("scalarMultMatrix(2,A))")
    showMatrix(ScalarMultMatrix(2,A))
    print("addMatrices(A,A)")
    showMatrix(AddMatrices(A,A))
    print("transpose(A)")
    showMatrix(Transpose(A))

###  uncomment next line to run initial tests 
#testMatrixFuncs()
testMatrix()


