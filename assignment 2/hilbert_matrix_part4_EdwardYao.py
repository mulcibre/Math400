from math import *

# creates zero matrix"
def zero(m, n):
    zero_matrix = [[0 for col in range(n)] for row in range(m)]
    return zero_matrix

# creates Hilbert b vector
def hilbert_b(n):
    b_vector = [0 for i in range(n)]
    b_vector[0] = 2/pi
    b_vector[1] = 1/pi
    for i in range(2,n):
        b_vector[i] = 1/pi -(i*(i-1)/pi**2)*b_vector[i-2]
    return(b_vector)

# creates Hilbert matrix
def hilbert(n):
    h_n = zero(n,n)
    for row in range(n):
        for col in range(n):
            h_n[col][row] = 1.0/(1.0+row+col)
    return(h_n)

# computes binomial coefficient nCk
def n_C_k(n,k):
    if k > n:
        n_ch_k = 0
    prod = 1.0
    for i in range(k):
        prod = prod*(1.0*(n-i)/(k-i))
    return(prod)

# creates inverse of a Hilbert matrix
def hilbert_inv(n):
    h_inv = zero(n,n)
    for k in range(n):
        for m in range(n):
            h_inv[k][m] = ((-1)**(k+m))*((k+m+1)*
                                          (n_C_k(n+k,n-m-1))*
                                          (n_C_k(n+m,n-k-1))*
                                          (n_C_k(k+m,k))**2)
    return(h_inv)

'''

print "Hilbert b Vector: \n", hilbert_b(3)
print "\nHilbert Matrix: \n", hilbert(3)
print "\nBinomial Coefficient: \n", n_C_k(3, 3)
print "\nHilbert Inverse: \n", hilbert_inv(3)

'''
