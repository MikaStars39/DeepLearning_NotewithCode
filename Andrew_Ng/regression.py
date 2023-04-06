import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_divergence, plt_gradients
plt.style.use('deeplearning.mplstyle')
# divergence 发散 gradients 梯度
x_train = np.array([1.0, 2.0])
# features
y_train = np.array([300.0, 500.0])
# target value


# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    cost = cost/(2 * m)
    return cost


# convergence 收敛
# 梯度下降的公式 the formula of gradient descent:
# J(w, b) = (1/2m)*sum(f(xi)-yi)^2)
# w = w - alpha * partial derivative of J(w, b) by w
# b = b - alpha * partial derivative of J(w, b) by b


def compute_gradient(x, y, w, b):
    m = len(x_train)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

    '''
    :param x: n lines array date examples
    :param y: n lines array target values
    :param w_in: scalar values of parameters
    :param b_in: scalar values of parameters
    :param alpha: float learning rate
    :param num_iters: int number of iterations to run gradient descent
    :param cost_function: the cost function
    :param gradient_function: the gradient function
    :return: w,b scalar the updated value; J_history list history of cost value p_history list history of parameters
    '''

    w = copy.deepcopy(w_in)
    # deepcopy means completely copy everything of one object
    # https://zhuanlan.zhihu.com/p/597973647

    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters/10) == 0:
                print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                      f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                      f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history

# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)

plt_divergence(p_hist, J_hist, x_train, y_train)
plt.show()
