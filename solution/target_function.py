import cv2
from scipy.ndimage import center_of_mass, distance_transform_edt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys, os

if __name__ == "__main__":
    from helper import SAMPLE_PATH
else:
    from .helper import SAMPLE_PATH


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def target_function(
    input: np.ndarray, 
    weight_com: float = .01, 
    exp_com: float = 0.5,
    weight_edge: float = .01,
    exp_edge: float = 1.5,
) -> np.ndarray:
    print(input)
    dist_from_edge = distance_transform_edt(1-input)
    edge_cost = ( weight_edge * dist_from_edge ) ** exp_edge
    
    com_x, com_y = center_of_mass(input)
    rows, cols = np.indices(np.shape(input))    
    dist_from_com = np.sqrt((rows - com_x)**2 + (cols - com_y)**2)
    com_cost = ( weight_com * dist_from_com ) ** exp_com

    return edge_cost + com_cost
    
if __name__ == "__main__":
    # ones = np.ones((100, 100))
    # test = np.zeros((300, 300))
    # test[100:200, 100:200] = 1

    test = cv2.imread(f"{SAMPLE_PATH}/reference21.png")
    test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    _, test = cv2.threshold(test, 1, 255, cv2.THRESH_BINARY)
    test = normalize(test)
    
    cost_func_res = target_function(test)

    grad = np.gradient(cost_func_res)
    grad_norm = np.linalg.norm(grad, axis=0)
    res = np.array(grad_norm/ np.max(grad_norm))
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(cost_func_res.shape[1])
    Y = np.arange(cost_func_res.shape[0])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, cost_func_res)
    plt.show()

    # cv2.namedWindow("CV2_test", cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow("CV2_test", 600, 600)
    # cv2.imshow("CV2_test", cost_func_res)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()
