import cv2
import numpy as np

from typing import Iterable

if __name__ == "__main__":
    from helper import DATA_PATH, find_parts
else:
    from .helper import DATA_PATH, find_parts


def show_imgs(images: Iterable[np.ndarray]) -> None:
    for ii, img in enumerate(images):
        cv2.imshow(f"{ii}", img.astype(np.uint8))
    cv2.waitKey(1)
    input()
    cv2.destroyAllWindows() 

def normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def grad(image: np.ndarray) -> np.ndarray:
    image = np.gradient(image)
    image = np.linalg.norm(image, axis=0)
    return np.array(image/ np.max(image))

def std_dev(image: np.ndarray, win_size: tuple[int]) -> np.ndarray:
    mean_loc_sq = cv2.filter2D(image, -1, np.ones(win_size))
    ret_val = np.sqrt((image-mean_loc_sq)**2)
    return 255 * ret_val / np.max(ret_val)

def abs_mean_dev(image: np.ndarray) -> np.ndarray:
    return np.abs(image - np.mean(image))

if __name__ == "__main__":
    part_id =1
    FILE_LIST = find_parts(DATA_PATH / f"part_{part_id}")
    NAME = FILE_LIST[3]
    print(f"LOADING: {NAME}")
    IMG = cv2.imread(NAME)

    IMG_RGB = IMG.copy()
    IMG_LAB = cv2.cvtColor(IMG, cv2.COLOR_BGR2LAB)
    IMG_HSV = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
    IMG_BW = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    
    l_channel, x, y = cv2.split(IMG_LAB)
    h, s, v = cv2.split(IMG_HSV)
    r, g, b, = cv2.split(IMG_RGB)
    
    win_size = (np.array(IMG_BW.shape) / 50).astype(np.int16)
    show_imgs([
        std_dev(x, win_size),
        std_dev(y, win_size),
        255* grad(IMG_BW),
        255* normalize(l_channel)
    ])