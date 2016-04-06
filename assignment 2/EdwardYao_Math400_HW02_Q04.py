import matplotlib.pyplot as plt
from math import *
from operator import mul

#Returns the products in a list
def listProduct(X):
    
    return reduce(mul, X)

#Returns Li(x)
def lagrange_i(nodes, x, i):
    
    index = range(len(nodes))
    index.remove(i) #makes sure denominator and numerator don't equal to 0
    denominator = listProduct([nodes[i] - nodes[j] for j in index])
    numerator = listProduct([x - nodes[j] for j in index])

    return numerator/denominator

#calculates lagrange
def lagrange(nodes, Y, x):
    
    N = len(nodes)
    result = sum([Y[i]*lagrange_i(nodes, x, 6) for i in range(N)])
    
    return result

#creates x,y table
def table(x, y, name):
    print "-------------------",name,"-------------------"      
    print "| \t # \t | \t X \t | \t Y \t |"
    for i in range (len(x)):
        print "| \t ", i, " \t | \t ", x[i], " \t | \t ", y[i], " \t |"

#plots graph      
def graph(x, y, neg_x, pos_x, neg_y, pos_y, name):
    plt.plot(x, y, color = "orange", label = name)
    plt.axis([neg_x, pos_x, neg_y, pos_y])
    plt.legend(loc = "lower left")
    plt.show()

#recursively calculates pi(x)
def pi_func(nodes, x, index):

    if index < 0:   # base case
        return 1
    else:
        result = (x - nodes[index]) * pi_func(nodes, x, index-1)  # recursive call
        #print result
        return result
            

#problem 4
def problem4():
    
    # number of plots for graph
    M = 100
    
    #####################################
    #### Lagrange Interpolation Test ####
    #####################################
    
    # nodes for lagrange interpolation
    nodes = [0, 1/24.0, 1/12.0, 1/6.0, 1/4.0, 1/3.0, 5/12.0, 1/2.0, 2/3.0, 5/6.0, 1.0]
    y = [2 * exp(-2 * x) * sin(3*pi*x) for x in nodes]
    
    # x values for interpolation
    x_values = [nodes[0] + (float(nodes[-1]) - nodes[0])/M * i for i in range(M + 1)]

    # y values for interpolation with Lagrange
    y_values = [lagrange(nodes, y, x) for x in x_values]
    
    ##########################
    #### Pi Function Test ####
    ##########################
    
    pi_nodes = [0, 1/10.0, 2/10.0, 3/10.0, 4/10.0, 5/10.0, 6/10.0, 7/10.0, 8/10.0, 9/10.0, 1.0]
    # x values for pi(x)
    pi_x = [pi_nodes[0] + (float(pi_nodes[-1]) - pi_nodes[0])/M * i for i in range(M + 1)]
    # y values for pi(x)
    pi_y = [pi_func(pi_nodes, x, len(pi_nodes)-1) for x in pi_x]
    
    
    ################
    #### Graphs ####
    ################
    
    # langrange graphs in different scales
    graph(x_values, y_values, 0, 1, -800, 100, "Lagrange")
    graph(x_values, y_values, 0, 1, -400, 50, "Lagrange")
    graph(x_values, y_values, 0, 1, -200, 25, "Lagrange")
    graph(x_values, y_values, 0, 1, -100, 15, "Lagrange")
    graph(x_values, y_values, 0, 1, -50, 10, "Lagrange")
    graph(x_values, y_values, 0, 1, -25, 5, "Lagrange")
    graph(x_values, y_values, 0, 1, -15, 4, "Lagrange")
    graph(x_values, y_values, 0, 1, -10, 3, "Lagrange")
    graph(x_values, y_values, 0, 1, -5, 2, "Lagrange")
    graph(x_values, y_values, 0, 1, -2, 1, "Lagrange")
    graph(x_values, y_values, 0, 1, -1, 1, "Lagrange")
    
    # pi(x) graphs in different scales
    graph(pi_x, pi_y, 0, 1, -0.000005, 0.000005, "Pi(x)")
    graph(pi_x, pi_y, 0, 1, -0.0000005, 0.0000005, "Pi(x)")
    graph(pi_x, pi_y, 0, 1, -0.00000005, 0.00000005, "Pi(x)")
    
    
    
    ################
    #### Tables ####
    ################
    
    print table(x_values,y_values, "Lagrange")
    print table(pi_x,pi_y, "Pi(x)")
    
    
    

problem4()