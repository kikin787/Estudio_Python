import numpy as np
import matplotlib.pyplot as plt

def rotate_2d(punto, angulo, center):
    """Rotates a 2D point around a center."""
    x, y = punto
    cx, cy = center
    angle_rad = np.radians(angulo)
    x_rotated = cx + np.cos(angle_rad) * (x - cx) - np.sin(angle_rad) * (y - cy)
    y_rotated = cy + np.sin(angle_rad) * (x - cx) + np.cos(angle_rad) * (y - cy)
    return x_rotated, y_rotated

def plot_figure_2d(points, rotated_points, center):
    """Plots 2D figure and its rotation."""
    plt.figure()
    plt.plot([p[0] for p in points], [p[1] for p in points], label='Original')
    plt.plot([p[0] for p in rotated_points], [p[1] for p in rotated_points], label='Rotated')
    plt.plot(center[0], center[1], 'ro', label='Rotation Center', color="purple")  # Plot rotation center in yellow
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Rotation')
    plt.legend()
    plt.axis('equal')
    plt.show()


# Define points for a 2D figure
points_2d_figure = [(-1, 1), (-1, -4), (-5, -4), (-5, 2), (0, 5), (5, 2), (5, -4), (1, -4), (1, 1), (-1, 1)]

# Define rotation center for 2D
center_2d = (0, 0)

# Example input
angle_2d = 90

# Rotate 2D figure
rotated_points_2d_figure = [rotate_2d(point, angle_2d, center_2d) for point in points_2d_figure]
plot_figure_2d(points_2d_figure, rotated_points_2d_figure, center_2d)
