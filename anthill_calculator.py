import numpy as np
from shapely.geometry import Point, Polygon


# returns the extremal points of the boundary.
def get_boundary_properties(boundary_input):
    boundary_size = len(boundary_input)
    temp_vector = [boundary_input[i][0] for i in range(boundary_size)]
    x_min_return, x_max_return = min(temp_vector), max(temp_vector)
    temp_vector = [boundary_input[i][1] for i in range(boundary_size)]
    y_min_return, y_max_return = min(temp_vector), max(temp_vector)
    return x_min_return, x_max_return, y_min_return, y_max_return


boundary = [[0, -4], [1, -4], [2, -3], [3, -2], [4, -1], [4, 0], [4, 1], [3, 2], [3, 3], [2, 4], [1, 5], [0, 5], [-1, 4], [-2, 3], [-3, 2], [-3, 1], [-3, 0], [-3, -1], [-3, -2], [-2, -3], [-1, -4]]

x_min, x_max, y_min, y_max = get_boundary_properties(boundary)

# finds the interior points.
interior = []
boundary_polygon = Polygon(boundary)
for y in range(y_min, y_max + 1):
    for x in range(x_min, x_max + 1):
        position = [x, y]
        point = Point(position)
        if boundary_polygon.contains(point):
            interior.append(position)

# creates a bijection between all the boundary/interior points and the integers 0, 1, ..., [total no. points] - 1.
# we need this to index the matrix elements later.
position_index = np.zeros((x_max - x_min + 1, y_max - y_min + 1)).astype(int)
index = 0
for y in range(y_min, y_max + 1):
    for x in range(x_min, x_max + 1):
        position = [x, y]
        if position in boundary or position in interior:
            position_index[position[0] - x_min][position[1] - y_min] = index
            index = index + 1
        else:
            position_index[position[0] - x_min][position[1] - y_min] = -1

# generates the system of linear equations which encodes the problem.
dimension = len(boundary) + len(interior)
A = np.zeros((dimension, dimension))
b = np.zeros(dimension)
for y in range(y_min, y_max + 1):
    for x in range(x_min, x_max + 1):
        position = [x, y]
        if position in boundary:
            A[position_index[x - x_min][y - y_min]][position_index[x - x_min][y - y_min]] = 1
        elif position in interior:
            A[position_index[x - x_min][y - y_min]][position_index[x - x_min][y - y_min]] = -1
            A[position_index[x - x_min][y - y_min]][position_index[x - x_min][y - 1 - y_min]] = 0.25
            A[position_index[x - x_min][y - y_min]][position_index[x - 1 - x_min][y - y_min]] = 0.25
            A[position_index[x - x_min][y - y_min]][position_index[x + 1 - x_min][y - y_min]] = 0.25
            A[position_index[x - x_min][y - y_min]][position_index[x - x_min][y + 1 - y_min]] = 0.25
            b[position_index[x - x_min][y - y_min]] = -1

# solves the system of linear equations.
t = np.linalg.solve(A, b)

print("Average time to reach boundary: {} seconds".format(t[position_index[-x_min][-y_min]]))