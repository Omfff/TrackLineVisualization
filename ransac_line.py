import random
import math
import numpy as np

def my_leastsq(x, y):
    """Fit line by least square method

    Args:
        x: X coordinate array of points
        y: Y coordinate array of points
    Returns:
        k: Slope of straight line
        b: Line intercept
    """
    # k numerator is the mean of xy times the mean of y
    # k denominator is the mean of the square of x minus the square of the mean of x
    k = ((x*y).mean() - x.mean()* y.mean())/(pow(x,2).mean()-pow(x.mean(),2))
    b = (pow(x,2).mean()*y.mean() - x.mean()* (x*y).mean())/(pow(x,2).mean()-pow(x.mean(),2))
    return k,b

def plot_fiting_result(x, y, x_in, y_in, k, b, t, T):
    """Plotting fitted line and other informaitons
    
    Args:
        x: X coordinate array of all points
        y: Y coordinate array of all points
        x_in: X coordinate array of inliers 
        y_in: Y coordinate array of inliers
        k: Slope of straight line
        b: Line intercept
        t: Distance threshold
        T: The proportion of inliers threshold
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_x = np.array(sorted(x))
    t = t * math.pow(k*k+1*1,0.5)

    ax.scatter(x, y, marker='o', alpha=0.5, s = 20, label='All points')

    ax.fill_between(sorted_x, k*sorted_x + b - t, k *
                    sorted_x + b + t,color = 'yellow', alpha=0.5, label="Inlier region")

    ax.scatter(x_in, y_in, marker='o', color='g',s= 20,alpha=0.5, label='Inliers')

    ax.plot(x, k*x + b, color='r',linewidth='2', label='Best fit line')

    ax.legend(loc='upper left')

    title_str = "Line of Best Fit: y = %.2fx + %.2f\n, Total points:%d  Inliers = %d  Threshold = %.2f %%" \
        % (k, b, len(x), len(x_in), T)

    ax.set_title(title_str)
    plt.tight_layout()
    plt.show()


def fit_line_by_ransac(point_list, sigma = 3, iters=50, T=0.8, isPlot = False):
    """Use RANSAC to fit line

    Args:
        point_list: 
        sigma: Distance threshold, the max distance between line and inliers.
        iters: Max iteration number
        T: It is a threshold which represents the proportion of inliers in all points. If the threshold is reached, the iteration
            will terminate.
        isPlot: bool value, to decide whether to draw a fitted line 
    Returns:
        [best_a, best_c] which means [Slope of straight line, Line intercept]
    """

    # Parameter estimation of the best model
    best_a = 0  # Slope of straight line
    best_b = 0  
    best_c = 0  # Line intercept
    best_inliers = []
    n_total = 0  # Inlier nums
    for ite in range(iters):
        # Randomly select two points to solve the model
        sample_index = random.sample(range(len(point_list)), 2)
        x_1 = point_list[1][sample_index[0]]
        y_1 = point_list[0][sample_index[0]]
        x_2 = point_list[1][sample_index[1]]
        y_2 = point_list[0][sample_index[1]]

        # ax + by + c = 0
        if x_2 == x_1:
            a = 1
            b = 0
            c = - x_1
        else:            
            a = (y_2 - y_1) / (x_2 - x_1)
            b = -1
            c = y_1 - a * x_1

        # Calculate the number of interior points
        total_inlier = 0
        dis= (abs(a*point_list[1]+b*point_list[0]+c))/(math.pow(a*a+b*b,0.5))
        inliers_list = [point_list[1][dis < sigma], point_list[0][dis < sigma]]
        total_inlier = len(inliers_list)

        # Judge whether the current model is better than the previously estimated model
        if total_inlier > n_total:
            n_total = total_inlier
            best_a = a
            best_b = b
            best_c = c
            best_inliers = inliers_list
            #print("New best line found:\nm=%0.3f, c=%0.3f" % (a,c))

        # Determine whether the current model has reached the threshold
        if total_inlier > T * len(point_list[0]):
            break

    # Re‐estimate the line using all points in best_inliers
    z = np.polyfit(best_inliers[0], best_inliers[1], 1) #my_leastsq(best_inliers[0], best_inliers[1])
    best_a = z[0]
    best_c = z[1]

    if isPlot:
        plot_fiting_result(point_list[1], point_list[0], best_inliers[0], best_inliers[1], best_a, best_c, sigma, T)
    
    return best_a,best_c
