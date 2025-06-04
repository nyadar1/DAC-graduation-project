import numpy as np


def get_traj(x):
    c_x = np.array([-6162.02, -3811, 262.2127187, 4699.890932, 3999.831872])
    c_y = np.array([-3889.57, -3267.91, -6119.909878, -3500.086201, -1000.110362])
    R = np.array([1493, 807, 3993, 307, 1007.2])
    x_begin = np.array([-6084.221, -4477.966, -2297.476, 4565.378, 5007.032])
    y_begin = np.array([-2398.598, -3722.227, -3055.266, -3776.049, -1000.060])
    x_end = np.array([-4920.867, -3297.543, 2019.163, 5006.891, 3999.988])
    y_end = np.array([-3059.758, -3890.494, -2534.220, -3500.258, 7.090])
    sign = np.array([1.0, -1.0, 1.0, -1.0])

    for i in range(4):
        if x >= x_begin[i] and x < x_end[i]:
            y = c_y[i] + sign[i] * (R[i] ** 2 - (x - c_x[i]) ** 2) ** 0.5
            psi = np.arctan2(y - c_y[i], x - c_x[i]) - 3.14159 / 2
            if sign[i] == -1:
                psi += 3.14159
            return y, psi
        if x >= x_end[i] and x < x_begin[i + 1]:
            k = (y_begin[i + 1] - y_end[i]) / (x_begin[i + 1] - x_end[i])
            y = y_end[i] + k * (x - x_end[i])
            psi = np.arctan(k)
            return y, psi