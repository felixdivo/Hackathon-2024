from functools import partial

import cv2
from scipy.ndimage import center_of_mass, distance_transform_edt, rotate
from scipy.optimize import minimize, OptimizeResult
import numpy as np
import matplotlib.pyplot as plt


SQRT_2 = np.sqrt(2)

if __name__ == "__main__":
    from helper import SAMPLE_PATH
else:
    from .helper import SAMPLE_PATH


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def create_target_map(
    input: np.ndarray,
    com_x: float,
    com_y: float,
    weight_com: float = 0.01,
    exp_com: float = 0.5,
    weight_edge: float = 0.01,
    exp_edge: float = 1.5,
) -> np.ndarray:
    dist_from_edge = distance_transform_edt(1 - input)
    edge_cost = (weight_edge * dist_from_edge) ** exp_edge

    rows, cols = np.indices(np.shape(input))
    dist_from_com = np.sqrt((rows - com_x) ** 2 + (cols - com_y) ** 2)
    com_cost = (weight_com * dist_from_com) ** exp_com

    return edge_cost + com_cost



def overlay_gripper(
    vars: np.ndarray,
    gripper: np.ndarray,
    base_img: np.ndarray,
) -> np.ndarray:
    x, y, alpha = vars
    gripper_mask = np.zeros(base_img.shape)
    gripper = gripper.astype(bool)
    gripper = rotate(gripper, alpha, reshape=True, order=0)
    gripper_h, gripper_w = gripper.shape
    gripper_cx, gripper_cy = gripper_w // 2, gripper_h // 2
    if gripper_mask.ndim == 2:
        gripper_mask[
            y - gripper_cy : y - gripper_cy + gripper_h,
            x - gripper_cx : x - gripper_cx + gripper_w,
        ] = gripper
    elif gripper_mask.ndim == 3:
        gripper_mask[
            :,
            y - gripper_cy : y - gripper_cy + gripper_h,
            x - gripper_cx : x - gripper_cx + gripper_w,
        ] = gripper
    else:
        raise NotImplemented()

    result_overlay = (
        base_img * gripper_mask
    )  # Could maybe be optimized for runtime

    return result_overlay


def target_function(
    vars: np.ndarray,
    gripper: np.ndarray,
    target_map: np.ndarray,
    grad_map: np.ndarray,
):
    result_overlay = overlay_gripper(np.floor(vars).astype(int), gripper, target_map)
    result_sum = np.sum(result_overlay)

    return result_sum


def target_grad(
    vars: np.ndarray,
    gripper: np.ndarray,
    target_map: np.ndarray,
    grad_map: np.ndarray,
):
    gripper_size = np.sum(gripper) / 100

    grad_overlay = overlay_gripper(np.floor(vars).astype(int), gripper, grad_map)
    total_force = np.sum(grad_overlay, axis=(1,2))

    x0, y0, alpha = vars

    x, y = np.meshgrid(np.arange(grad_overlay.shape[2]), np.arange(grad_overlay.shape[1]))
    rx = x - x0
    ry = y - y0

    torque = rx * grad_overlay[0,:,:] - ry * grad_overlay[1,:,:]
    gripper_r = np.max(gripper.shape)/2
    total_torque = np.sum(torque) / gripper_r

    total = np.concatenate((total_force, [total_torque]))
    print(total)
    return total



def show_res(gripper, target_map, res):
    fig, ax = plt.subplots(2)
    ax[0].contourf(target_map, vmin=0, vmax=10)
    out = overlay_gripper(np.floor(np.array([res[0], res[1], res[2]])).astype(int), gripper, target_map)
    ax[1].contourf(out, vmin=0, vmax=10)
    plt.show()


def optimize(
    gripper: np.ndarray, target_map: np.ndarray, grad_map: np.ndarray, com_x: int, com_y: int
) -> OptimizeResult:
    gripper_r = np.max(gripper.shape)/2
    gripper_diag = gripper_r*SQRT_2
    rot_step = np.atan(1/gripper_r) * 2**7
    res = minimize(
        target_function,
        np.array([100, 1000, 0]),
        (gripper, target_map, grad_map),
        "SLSQP",
        jac=target_grad,
        # callback=partial(show_res, gripper, target_map),
        bounds=[(gripper_diag,target_map.shape[1]-gripper_diag),(gripper_diag,target_map.shape[0]-gripper_diag),(-180, 180)],
        options={"disp": True, "eps": (1, 1, rot_step)},
    )
    return res


if __name__ == "__main__":
    part = cv2.imread(f"{SAMPLE_PATH}/reference21.png")
    part = cv2.cvtColor(part, cv2.COLOR_RGB2GRAY)
    _, part = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
    part = normalize(part)

    gripper = cv2.imread(f"{SAMPLE_PATH}/gripper_2.png")
    gripper = cv2.cvtColor(gripper, cv2.COLOR_RGB2GRAY)
    gripper[gripper > 0] = 1

    com_x, com_y = center_of_mass(part)
    target_map = create_target_map(part, com_x, com_y)
    grad_map = np.stack(np.gradient(target_map))


    res = optimize(gripper, target_map, grad_map, com_x, com_y)
    show_res(gripper, target_map, res.x)
