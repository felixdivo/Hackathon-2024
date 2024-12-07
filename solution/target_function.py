import cv2
from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path
import sys, os

SCRIPT_PATH: Path = Path(os.path.abspath(sys.argv[0])).parent
PROJECT_PATH: Path = SCRIPT_PATH.parent
SAMPLE_PATH: Path = PROJECT_PATH / "TEST_EXAMPLES"


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def part_cog(input: np.ndarray) -> tuple[float, float]:
    total_ones = np.sum(input == 1)
    x_dist = np.sum(input, axis=0)

    mean_x = np.sum(
        [ii * x_dist[ii] for ii in range(input.shape[1])]
    ) / total_ones

    y_dist = np.sum(input, axis=1)
    mean_y = np.sum(
        [ii * y_dist[ii] for ii in range(input.shape[0])]
    ) / total_ones

    return mean_x, mean_y

def target_function(
    input: np.ndarray, 
    weight_rad: float = 1., 
    weight_edge: float = 1.,
    sigma_edge: float = 20.
) -> np.ndarray:
    print(input)
    edge_cost_func = 1- normalize(gaussian_filter(input, sigma_edge))
    
    mean_x, mean_y = part_cog(input)
    r_func = lambda x, y: (x-mean_x)**2 + (y-mean_y)**2
    center_cost_func = normalize(np.fromfunction(r_func, input.shape))

    print(mean_x, mean_y)

    return normalize(
        weight_edge*edge_cost_func + weight_rad*center_cost_func
    )
    
if __name__ == "__main__":
    # ones = np.ones((100, 100))
    # test = np.zeros((300, 300))
    # test[100:200, 100:200] = 1

    test = cv2.imread(f"{SAMPLE_PATH}/reference21.png")
    test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    _, test = cv2.threshold(test, 1, 255, cv2.THRESH_BINARY)
    test = normalize(test)
    
    cost_func_res = target_function(test, 10)

    grad = np.gradient(cost_func_res)
    grad_norm = np.linalg.norm(grad, axis=0)
    res = np.array(grad_norm/ np.max(grad_norm))
    
    cv2.imshow("2D Array", res)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
