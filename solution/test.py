from functools import partial

import cv2
from scipy.ndimage import center_of_mass, distance_transform_edt, rotate
from scipy.optimize import minimize, OptimizeResult
import numpy as np


if __name__ == "__main__":
    from helper import SAMPLE_PATH
else:
    from .helper import SAMPLE_PATH, PROJECT_PATH

def normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def gripper_mask(
    vars: np.ndarray,
    gripper: np.ndarray,
    base_img: np.ndarray,
) -> np.ndarray:
    x, y, alpha = vars
    gripper_mask = np.zeros(base_img.shape, dtype=bool)
    gripper = gripper.astype(bool)
    gripper = rotate(gripper, alpha, reshape=True, order=0)
    gripper_h, gripper_w = gripper.shape
    gripper_cx, gripper_cy = gripper_w // 2, gripper_h // 2
    match gripper_mask.ndim:
        case 2:
            gripper_mask[
                y - gripper_cy : y - gripper_cy + gripper_h,
                x - gripper_cx : x - gripper_cx + gripper_w,
            ] = gripper
        case 3:
            gripper_mask[
                :,
                y - gripper_cy : y - gripper_cy + gripper_h,
                x - gripper_cx : x - gripper_cx + gripper_w,
            ] = gripper
        case _:
            raise NotImplemented()
    return gripper_mask

def create_target_cost_func(
    input: np.ndarray,
    com_x: float,
    com_y: float,
    weight_com: float = 3,
    exp_com: float = 1,
    weight_edge: float = 0.001,
    exp_edge: float = 10,
) -> np.ndarray:
    # dist_from_edge = distance_transform_edt(1 - input) 
    # edge_cost = dist_from_edge ** exp_edge
    # edge_cost *= (weight_edge / np.max(edge_cost)) 

    # rows, cols = np.indices(np.shape(input))
    # dist_from_com = np.sqrt((cols - com_x) ** 2 + (rows - com_y) ** 2)
    # com_cost = dist_from_com ** exp_com
    # com_cost *= (weight_com / np.max(com_cost))

    dist_from_edge = distance_transform_edt(1 - input)
    edge_cost = (weight_edge * dist_from_edge) ** exp_edge

    rows, cols = np.indices(np.shape(input))
    dist_from_com = np.sqrt((cols - com_x) ** 2 + (rows - com_y) ** 2)
    com_cost = (weight_com * dist_from_com) ** exp_com

    return edge_cost + com_cost
    
def I_p(gripper: np.ndarray) -> float:
    rows, cols = np.nonzero(gripper)
    x_c = cols.mean()
    y_c = rows.mean()
    dx = cols - x_c
    dy = rows - y_c

    # Polar moment of inertia
    return np.sum(dx**2 + dy**2) 

def rot_inc(
    X_grad: np.ndarray, 
    Y_grad: np.ndarray, 
    mask: np.ndarray, 
    gripper_x: int, 
    gripper_y: int,
    I_gripper: float
) -> float:
    y_coords, x_coords = np.indices(mask.shape)
    # Calculate the momentum using vectorized operations
    momentum = (gripper_x - x_coords) * Y_grad - (gripper_y - y_coords) * X_grad
    return np.sum(momentum) / I_gripper

def trans_inc(X_grad: np.ndarray, Y_grad: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    return (
        # np.max(1, int(np.sum(X_grad[mask]) / np.sum(mask))),
        # np.max(1, int(np.sum(Y_grad[mask]) / np.sum(mask))) 
        int(np.sum(X_grad[mask])),
        int(np.sum(Y_grad[mask])),
    )

def err_func(
    com_part: tuple[int, int], 
    gripper_xy: tuple[int, int], 
    cost_function: np.ndarray,
    mask: np.ndarray, 
    I_gripper: float
) -> float:
    com_x, com_y = com_part
    g_x, g_y = gripper_xy
    return np.sqrt((g_x - com_x)**2 + (g_y - com_y)**2) * np.sum(cost_function[mask]) / I_gripper


if __name__ == "__main__":
    ### TESTING ###
    # grad_x = np.zeros((3,3))
    # grad_x[0,2] = 1
    # grad_y = grad_x.copy()
    # grad_x[2,1] = -1
    # print(rot_inc(grad_x, grad_y, np.ones((3,3), dtype=bool), 1, 1))
    # print(trans_inc(grad_x, grad_y, np.ones((3,3), dtype=bool)))
    ### ENDE ###
    
    part = cv2.imread(f"{SAMPLE_PATH}/reference21.png")
    part = cv2.cvtColor(part, cv2.COLOR_RGB2GRAY)
    part = cv2.copyMakeBorder(part, 300, 300, 300, 300, cv2.BORDER_CONSTANT)
    
    _, part = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
    part = normalize(part)

    gripper = cv2.imread(f"{SAMPLE_PATH}/gripper_2.png")
    gripper = cv2.cvtColor(gripper, cv2.COLOR_RGB2GRAY)
    gripper[gripper > 0] = 1
    I_gripper = I_p(gripper)

    com_x, com_y = center_of_mass(part)
    COST_FUNC = create_target_cost_func(part, com_x, com_y)
    GRAD_COST_FUNC = np.stack(np.gradient(COST_FUNC))
    X_grad = GRAD_COST_FUNC[0]
    Y_grad = GRAD_COST_FUNC[1]

    X_pos, Y_pos, rot = 2000, 300, 0
    err = 0
    for iter_ in range(31):
        # print(X_pos, Y_pos, rot)
        mask = gripper_mask((round(X_pos, 0), round(Y_pos, 0), rot), gripper, part)
        dx, dy = trans_inc(X_grad, Y_grad, mask)
        da = rot_inc(X_grad, Y_grad, mask, X_pos, Y_pos, I_gripper)
        dx = np.sign(dx) * np.min((np.abs(dx), X_pos/3, (part.shape[1] - X_pos)/3))
        dy = np.sign(dy) * np.min((np.abs(dy), Y_pos/3, (part.shape[0] - Y_pos)/3))
        print(f"Iter {iter_} dX: {dx}, dY: {dy}, dAng: {da:.3f}, Errorfx: {err:.3f}")
        
        err_weight =  .3#* err

        X_pos += np.max((int(err_weight  * dx), 1))
        Y_pos += np.max((int(err_weight  * dy), 1))
        rot +=  err_weight * da
        err = err_func((com_x, com_y), (X_pos, Y_pos), COST_FUNC, mask, I_gripper)
        # cv2.imwrite(SAMPLE_PATH / f"optimization/optimization_{iter_}.png", normalize(part+mask))

    print(f"FINAL ITERATION RESULTS: dX: {dx}, dY: {dy}, dAng: {da}, Errorfx: {err}")
    cv2.imshow("IMAGES", cv2.resize(normalize(part+mask), (500, 300))); 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 