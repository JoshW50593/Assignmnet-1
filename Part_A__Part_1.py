import numpy as np
import matplotlib.pyplot as plt



beta = 11.1
#deriving the least swuares error, set too zero, solve for wights

def x_data_points(data):
    x = np.linspace(0,1,len(data))
    return x

def load_data_1():
    data_1 = np.loadtxt("Dataset_1.txt")

    return data_1

def load_data_2():
    data_2 = np.loadtxt("Dataset_2.txt")

    return data_2

def design_matrix_func(x, M):
    design_matrix = []

    for i in x:  # the number of columns
        basis_function = []
        for j in range(M + 1):  # the number of columns. the order of the polynomial starts at zero but goes to the number stateed for M
            basis_function.append(i ** j)  # (xi)^j and represnets a row
        design_matrix.append(basis_function)
    return design_matrix

def least_squares(x, data, M):
    #we want to determin mininal w
    #t = load_data
    design_matrix = design_matrix_func(x, M)

     # this would be the basis function for one enitre x dataset and would form a coulm in the design matrix]
    inv_dot = np.linalg.inv(np.dot(np.transpose(design_matrix),design_matrix))
    design_data_dot = np.dot(np.transpose(design_matrix), data)
    W_ml = np.dot(inv_dot, design_data_dot)
    
    x1 = np.linspace(0,1,10)
    y1=[]
    x2 = np.linspace(0,1,50)
    y2=[]
    x3 = np.linspace(0,1,99)
    y3=[]
    index=0
    
    for i in x1:
        y_data_point=0
        for j in range(M+1):
            y_data_point += W_ml[j]*(i)**j
        y1.append(y_data_point)

    for i in x2:
        y_data_point=0
        for j in range(M+1):
            y_data_point += W_ml[j]*(i)**j
        y2.append(y_data_point)

    for i in x3:
        y_data_point=0
        for j in range(M+1):
            y_data_point += W_ml[j]*(i)**j
        y3.append(y_data_point)

    std_1 = np.std(y1)
    std_2 = np.std(y2)
    std_3 = np.std(y3)
    # create a figure with 3 subplots in a 1x3 grid
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))


    # plot the first subplot
    axs[0].plot(x1, y1)
    axs[0].plot(x1, data, "x")
    axs[0].fill_between(x1, y1 - std_1, y1 + std_1, alpha=0.25)
    axs[0].set_title("Plot with 10 datapoints")

    # plot the second subplot
    axs[1].plot(x2, y2)
    axs[1].plot(x1, data, "x")
    axs[1].fill_between(x2, y2 - std_2, y2 + std_2, alpha=0.25)
    axs[1].set_title("Plot with 50 datapoints")

    # plot the third subplot
    axs[2].plot(x3, y3)
    axs[2].plot(x1, data, "x")
    axs[2].fill_between(x3, y3 - std_3, y3 + std_3, alpha=0.25)
    axs[2].set_title("Plot with 100 datapoints")

    # add a title to the entire figure


    # show the figure
    plt.show()

    plt.show()
    print(W_ml)
    return W_ml
    # 
     #is a mtrix of basis functions, dat points are stored in the columns
    #each row is a basis function for a given data point
     

def Bayesian(x, data_1, M):
    m0=0

    alpha = 5e-3
    variance = 1/alpha

    design_matrix = design_matrix_func(x, M)
    I = np.eye(M+1)

    inv_dot = np.linalg.inv(np.dot(np.transpose(design_matrix),design_matrix))
    design_data_dot = np.dot(np.transpose(design_matrix), data_1)
    W_ml = np.dot(inv_dot, design_data_dot)

    Sn_inv = alpha*I+beta*np.dot(np.transpose(design_matrix),design_matrix)

    Sn = np.linalg.inv(Sn_inv)
    m_N_1 =  np.dot(beta,Sn)
    m_N_2 = np.dot(np.transpose(design_matrix), data_1)
    m_N = np.dot(m_N_1, m_N_2)
    print(m_N)
    y1 = []
    variance_pred_dis = 0
    for i in x:
        var_temp = 0
        basis_func = []
        for j in range(M+1):
            basis_func.append(i ** j)
        var_t_1 = np.dot(Sn, (basis_func))
        variance_pred_dis = 1/beta + np.dot(np.transpose(basis_func), var_t_1)
        y1.append(np.dot(np.transpose(m_N), basis_func)) #this is the expcted value as it is the mean




    plt.figure()
    plt.plot(x,y1, "red")
    plt.plot(x, data_1, "x")
    plt.fill_between(x , y1-np.sqrt(variance_pred_dis), y1+np.sqrt(variance_pred_dis), alpha=0.25)
    plt.show()

    return y1, variance_pred_dis


M=4
data_1 = load_data_1()
x_data =x_data_points(data_1)
data_2 = load_data_2()
least_squares(x_data, data_1, M)
Bayesian(x_data, data_1, M)
#design_matrix_func(x_data)

#y_data_points(M, x_data)

#w_ml = least_squares(x_data, data_1, 4)
#print(w_ml)
#Bayesian(x_data,data_1, 5)