#bayesian model selection
#this is done on dataset 2
#model evidence = marginal likeliyhood

import numpy as np


import matplotlib.pyplot as plt

# import Part_A__Part_1 as p
import Part_A__Part_1 as pAp1

beta = 11.1
alpha = 5e-3

def x_data_points(data):
    x = np.linspace(0,1,len(data))
    return x

def load_data_2():
    data_2 = np.loadtxt("Dataset_2.txt")
    return data_2

def model_order(M_in):
    M = M_in
    return M

def design_matrix_func(x, M):
    design_matrix = []

    for i in x:  # the number of columns
        basis_function = []
        for j in range(M + 1):  # the number of columns. the order of the polynomial starts at zero but goes to the number stateed for M
            basis_function.append(i ** j)  # (xi)^j and represnets a row
        design_matrix.append(basis_function)
    return design_matrix

def A_matrix(M,d_m):
    I = np.eye(M+1)
    aaa = beta * np.dot(np.transpose(d_m), d_m)
    A = alpha * I + beta * np.dot(np.transpose(d_m), d_m) #Sn_inv = A

    return A

def M_n_func(data, M, d_m):
    A = A_matrix(M, d_m)#Sn_inv = A
    A_inv = np.linalg.inv(A) #A_inv = Sn

    M_n_1 = np.dot(np.transpose(d_m), data)
    M_n_2 = np.dot(A_inv, M_n_1)
    M_n = beta*M_n_2

    return M_n

def E_Mn_func(beta, alpha, data, d_m, M_n):
    E_mn_1 = (beta/2)*(np.linalg.norm(data - np.dot(d_m, M_n)))**2
    E_mn_2 = (alpha/2)*(np.dot(np.transpose(M_n), M_n))
    E_Mn = E_mn_1+E_mn_2

    return E_Mn

def bayesian_model_selection(x, data, M, alpha, beta):
    ab =0

    ln_pred = []
    pred= []
    M_arr=[]


    for i in range(M+1):
        M_arr.append(i)
        design_matrix = design_matrix_func(x, i)
        A = A_matrix(i, design_matrix)
        M_n = M_n_func(data, i, design_matrix)
        E_mn = E_Mn_func(beta, alpha, data, design_matrix, M_n)
        ln_pred_at_M = (i/2)*np.log(alpha) + (len(data)/2)*np.log(beta) - E_mn - 0.5*np.log(np.linalg.det(A)) - (len(data)/2)*np.log(2*np.pi)
        ln_pred.append(ln_pred_at_M)

    pred = np.exp(ln_pred)
    plt.figure()
    plt.plot(M_arr, pred)
    plt.xlabel("Model Order")
    plt.ylabel("Marginal likelihood")

    plt.figure()
    plt.plot(M_arr, ln_pred)

    plt.show()

    return pred

def bayesian_model_averaging(pred_values, M, data, x):
#1 find area under marginal likihood
#2 normalize by dividng each M point by the sum
#3 multiply by M_n*basis_function for each M
#4 plot
    P_m = np.zeros(len(x))
    pred_sum = np.sum(pred_values)
    #print(pred_sum)
    total_sum=0
    pred_graph=[]
    M_arr=[]
    #print(len(pred_values))
    for i in range(M+1):
        bayesian_mean, bayesian_var = pAp1.Bayesian(x, data, i)
        bayesian_std = np.sqrt(bayesian_var)
        P_i = pred_values[i]/pred_sum
        if i==3:
            print("enteed")
            plt.figure()
            plt.plot(x, bayesian_mean, "red")
            plt.fill_between(x, bayesian_mean - bayesian_std, bayesian_mean + bayesian_std, alpha=0.25)
            plt.xlabel("Predictor variable")
            plt.ylabel("Response variable")
            #plt.title("Plot of mean and standard deviation of model order 4")
            plt.show()
        total_sum+=P_i
        print(total_sum)
        P_m += np.dot(bayesian_mean, P_i)
        std = np.dot(bayesian_std, P_i)


    plt.figure()
    plt.plot(x, P_m, "black")
    plt.fill_between(x, P_m-std, P_m+std, alpha=0.5)
    plt.xlabel("Predictor variable")
    plt.ylabel("Response variable")
    #plt.title("P Averaged Model ")
    plt.show()

    # plt.figure()
    # plt.plot(x,pred_values, "red")
    # plt.plot(x, data, "x")
    # plt.fill_between(x , pred_values-std, pred_values+std, alpha=0.25)
    # plt.title("Plot of scaled values")
    # plt.show()






data_2 = load_data_2()
x = x_data_points(data_2)

M=9

pred_v = bayesian_model_selection(x, data_2, M, alpha, beta)
bayesian_model_averaging(pred_v, M, data_2, x)
