import cv2
from scipy.ndimage import gaussian_filter
import numpy as np

def target_function(input):
    print(input)
    output = gaussian_filter(input, 10)
    grad = np.gradient(output)
    grad_norm = np.linalg.norm(grad, axis=0)
    res = grad_norm / np.max(grad_norm)
    cv2.imshow("2D Array", ~res)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ones = np.ones((100, 100))
    test = np.zeros((300, 300))
    test[100:200, 100:200] = ones
    target_function(test)