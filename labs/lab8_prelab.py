import numpy as np

sample_points = np.linspace(0, 10, 100)
print(sample_points)


def get_sub_interval_indexes(a: float, b: float, points: []) -> []:
    above = np.where(points >= a)[0]
    below = np.where(points <= b)[0]
    return list(set(above) & set(below))


indexes = get_sub_interval_indexes(0, 2, sample_points)

# some testing
# for i in indexes:
#     print('i = ', i)
#     print('--', x[i])


def line_through(x_0: float, x_1: float, f_0: float, f_1: float) -> [(float, float)]:
    return lambda x: (f_1 - f_0) / (x_1 - x_0) * (x - x_0) + f_0
