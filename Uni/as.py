import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def rotacion_3D(punto, angulo, centro):
    """Rotates a 3D point around a center."""
    x, y, z = punto
    cx, cy, cz = centro
    angulo_rad = np.radians(angulo)
    rotacion_matriz = np.array([[np.cos(angulo_rad), -np.sin(angulo_rad), 0],
                                [np.sin(angulo_rad), np.cos(angulo_rad), 0],
                                [0, 0, 1]])
    rotacion_punto = np.dot(rotacion_matriz, np.array([x - cx, y - cy, z - cz])) + np.array([cx, cy, cz])
    return rotacion_punto

def plot_figura_3D(puntos, rotacion_puntos, centro):
    """Plots 3D figure and its rotation."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.plot([p[0] for p in puntos], [p[1] for p in puntos], [p[2] for p in puntos], label='Original', color='blue')
    
    # Plot rotated points
    ax.plot([p[0] for p in rotacion_puntos], [p[1] for p in rotacion_puntos], [p[2] for p in rotacion_puntos], label='Rotacion', color='red')
    
    # Plot rotation center
    ax.scatter(centro[0], centro[1], centro[2], label='Centro de Rotacion', color='yellow')
    
    # Connect points in parallel at the same level
    for i in range(0, len(puntos)//2):
        if puntos[i] != (-1, 4, 0) and puntos[i+len(puntos)//2] != (-1, 2, 0.5):
            ax.plot([puntos[i][0], puntos[i+len(puntos)//2][0]], 
                    [puntos[i][1], puntos[i+len(puntos)//2][1]], 
                    [puntos[i][2], puntos[i+len(puntos)//2][2]], color='blue')
            ax.plot([rotacion_puntos[i][0], rotacion_puntos[i+len(puntos)//2][0]], 
                    [rotacion_puntos[i][1], rotacion_puntos[i+len(puntos)//2][1]], 
                    [rotacion_puntos[i][2], rotacion_puntos[i+len(puntos)//2][2]], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotación 3D')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=2)
    plt.show()

# Define points for a 3D figure
puntos_figura_3D = [
    (-1, 1, 0), (-1, -4, 0), (-5, -4, 0), (-5, 2, 0), (0, 5, 0), (5, 2, 0), (5, -4, 0), (1, -4, 0), (1, 1, 0),
    (-1, 1, 0.5), (-1, -4, 0.5), (-5, -4, 0.5), (-5, 2, 0.5), (0, 5, 0.5), (5, 2, 0.5), (5, -4, 0.5), (1, -4, 0.5), (1, 1, 0.5)
]

# Define rotation center for 3D
centro_3D = (0, 0, 0)

# Example input
angulo_3D = 90

# Rotate 3D figure
puntos_rotados_figura_3D = [rotacion_3D(punto, angulo_3D, centro_3D) for punto in puntos_figura_3D]
plot_figura_3D(puntos_figura_3D, puntos_rotados_figura_3D, centro_3D)
