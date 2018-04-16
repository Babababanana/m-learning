import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_cost(errors):
    m = len(errors)
    cost = 0
    for err in errors:
        cost += err ** 2

    return cost / (2 * m)


def compute_cost(tr_x, tr_y, theta_0, theta_1):
    m = len(tr_x)
    errors = theta_0 * np.ones(m) + theta_1 * tr_x - tr_y
    return get_cost(errors)


def batch_gradient_desc(tr_x, tr_y, theta_0, theta_1, alpha, num_iter=1500):
    m = len(tr_x)

    for i in range(num_iter):
        # Calculate errors
        hypo = theta_0 * np.ones(m) + theta_1 * tr_x
        errors = hypo - tr_y

        # Update theta
        temp0 = theta_0 - alpha * np.sum(errors) / m
        temp1 = theta_1 - alpha * np.sum(errors * tr_x) / m
        theta_0 = temp0
        theta_1 = temp1

    return [theta_0, theta_1]


def linear_reg(train_x, train_y):
    # Initial settings
    theta_0 = theta_1 = 0

    alpha = 0.01
    theta = batch_gradient_desc(train_x, train_y, theta_0, theta_1, alpha)

    print(theta)

    # Visualization
    # Plot cost function
    cost_fig = plt.figure()
    ax_cost = cost_fig.add_subplot(111, projection='3d')
    ax_theta0 = ax_theta1 = np.linspace(-10, 10, 100)
    AX_THETA0, AX_THETA1 = np.meshgrid(ax_theta0, ax_theta1)
    costs = np.array([compute_cost(train_x, train_y, theta0, theta1) for theta0, theta1, in zip(np.ravel(AX_THETA0), np.ravel(AX_THETA1))])
    COST = costs.reshape(AX_THETA0.shape)

    ax_cost.plot_surface(AX_THETA0, AX_THETA1, COST)

    ax_cost.set_title('Cost function')
    ax_cost.set_xlabel('THETA_0')
    ax_cost.set_ylabel('THETA_1')
    ax_cost.set_zlabel('COST')

    # Plot training data and fitting line
    fitting_fig, ax_line = plt.subplots()
    hypo = theta[0] * np.ones(len(train_x)) + theta[1] * train_x
    ax_line.scatter(train_x, train_y)
    ax_line.plot(train_x, hypo)

    ax_line.set_title('Linear regression')
    ax_line.set_xlabel('X')
    ax_line.set_ylabel('Y')

    plt.show()

df = pd.read_csv('data.csv')
X = df.iloc[:, 0]
y = df.iloc[:, -1]
linear_reg(X, y)
