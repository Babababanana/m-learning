import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# generate N uniformly distributed [-1, 1]*[-1, 1] data points
def generate_points(points_N):
    data_points = np.random.uniform(low=-1, high=1, size=(points_N, 2))
    y = np.sign([point[1] - polynomial(point[0]) for point in data_points])
    df = pd.DataFrame(data=data_points, columns=['x1', 'x2'])
    df['y'] = y
    return df

# pick a target function
def pick_f():
    f_points = np.random.uniform(low=-1, high=1, size=(2, 2))
    coefficients = np.polyfit([f_points[0][0], f_points[1][0]],
                              [f_points[0][1], f_points[1][1]], 1)
    f = np.poly1d(coefficients)
    return f

# for one set of w, return the indexes of all the misclassified points
def get_misclassified(w, df):
    cnt_row = df.shape[0]
    hypo_y = []
    for i in range(cnt_row):
        hypo_y.append(np.sign(w.transpose()
                                  @ np.array([[1],
                                              [df.at[i, 'x1']],
                                              [df.at[i, 'x2']]])))

    return [i for i in range(df.shape[0])
                     if hypo_y[i] != df.at[i]['y']]

def pla(df):
    w = np.array([[float(0)],
                  [float(0)],
                  [float(0)]])
    
    misclassified = get_misclassified(w, data)
    iteration = 0
    while(misclassified and iteration < 1000):
        mis_index = misclassified[randint(0, len(misclassified) - 1)]
        w += df.at[mis_index, 'y'] * np.array([[1],
                                               [df.at[mis_index, 'x1'],
                                               [df.at[mis_index, 'x2']]])
        
                                              
    
    pass

def run_experiment(run_times, points_N):
    pass
