import numpy as np
import matplotlib.pyplot as plt

# matplotlib is a popular library for plotting data
# .pyplot allows matplotlib to draw lines and graphs like Matlab
plt.style.use('deeplearning.mplstyle')
# choose a style
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
# create the training set
print('x_train is {}'.format(x_train))
print('y_train is {}'.format(y_train))

print(f"x_train.shape: {x_train.shape}")
m0 = x_train.shape[0]
# or
# m = len(x_train)
print(f"x_train.shape: {m0}")

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title('Housing Price')
plt.ylabel('Price / 1000 dollar')
plt.xlabel('Size / 1000 sqr')
plt.show()

# f = wx + b

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")


def compute_model_output(x, w0, b0):
    m = len(x_train)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w0 * i + b0

    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
