import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
    
from typing import Iterable

if __name__ == "__main__":
    from helper import DATA_PATH, find_parts
else:
    from .helper import DATA_PATH, find_parts

def __optimal_grid(n: float) -> tuple[int, int]:
    sqrt_n = np.sqrt(n)
    rows = np.floor(sqrt_n)
    cols = np.ceil(sqrt_n)
    
    while rows * cols < n:
        if cols - rows > 1:
            rows += 1
        else:
            cols += 1

    return int(rows), int(cols)

def show_imgs(images: dict[str, np.ndarray]) -> None:
    n_img = len(images)
    grid_r, grid_c = __optimal_grid(n_img)
    n_fill = grid_r * grid_c - n_img

    for _,item in images.items():
        img_h, img_w = item.shape # fuck it
        break

    grid = np.zeros((grid_r * img_h, grid_c * img_w, 3), dtype=np.uint8)
    
    for idx, (label, img) in enumerate(images.items()):
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img = cv2.putText(
            img=img, 
            text=label, 
            org=(int(img.shape[0]/15), int(img.shape[1]/5)), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=.7, 
            color=(0,0,255), 
            thickness=2, 
            lineType=cv2.LINE_AA, 
            bottomLeftOrigin=False
        )
        row, col = divmod(idx, grid_c)
        grid[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w, :] = img
    
    cv2.imshow("IMAGES", grid)
    cv2.waitKey(1)
    input()
    cv2.destroyAllWindows() 

def normalize(image: np.ndarray) -> np.ndarray:
    return (255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))).astype(np.uint8)

def grad(image: np.ndarray) -> np.ndarray:
    image = np.gradient(image)
    image = np.linalg.norm(image, axis=0)
    return (255 * image/ np.max(image)).astype(np.uint8)

def std_dev(image: np.ndarray, win_size: tuple[int]) -> np.ndarray:
    mean_loc_sq = cv2.filter2D(image, -1, np.ones(win_size))
    ret_val: np.ndarray = np.sqrt((image-mean_loc_sq)**2)
    return (255 * ret_val / np.max(ret_val)).astype(np.uint8)

def abs_mean_dev(image: np.ndarray) -> np.ndarray:
    return np.abs(image - np.mean(image))

def smooth(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5,5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # sigma = np.min(image.shape) / 100
    # return gaussian_filter(image, sigma)
    return image
    
def canny_edge(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    ret = normalize(clahe.apply(image))
    ret =  cv2.Canny(ret, 150, 150, L2gradient=True)
    kernel = np.ones((5,5))
    ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((1,1))
    # ret = cv2.morphologyEx(ret, cv2.MORPH_ERODE, kernel)
    return ret

def clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    return normalize(clahe.apply(image))
    
def fill_edge(edge: np.ndarray) -> np.ndarray:
    _, edge = cv2.threshold(edge, 200, 255, cv2.THRESH_BINARY)        
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea)
    h, w = edge.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)  # Add a 1-pixel border around the image

    # Seed point inside the largest contour
    # (You can use cv2.boundingRect to find a seed point)
    x, y, w, h = cv2.boundingRect(largest_contour)
    seed_point = (x + w // 2, y + h // 2)

    # Flood-fill the largest contour
    filled_image = edge.copy()  # Make a copy of the binary image
    cv2.floodFill(filled_image, mask, seed_point, 255)
    return filled_image

if __name__ == "__main__":
    part_id = 4
    FILE_LIST = find_parts(DATA_PATH / f"part_{part_id}")
    NAME = FILE_LIST[3]
    print(f"LOADING: {NAME}")
    IMG = cv2.imread(NAME)
    b_size = 20
    IMG = cv2.copyMakeBorder(IMG, b_size, b_size, b_size, b_size, cv2.BORDER_CONSTANT)

    IMG_RGB = IMG.copy()
    IMG_LAB = cv2.cvtColor(IMG, cv2.COLOR_BGR2LAB)
    IMG_HSV = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
    IMG_BW = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    
    l_channel, x, y = cv2.split(IMG_LAB)
    h, s, v = cv2.split(IMG_HSV)
    r, g, b, = cv2.split(IMG_RGB)
    
    win_size = (np.array(IMG_BW.shape) / 50).astype(np.int16)
    
    TEST = np.clip(
        canny_edge(l_channel) +canny_edge(s) + canny_edge(v) + canny_edge(std_dev(x, win_size)) + canny_edge(std_dev(x, win_size)),#+ canny_edge(std_dev(r, win_size))+canny_edge(std_dev(g, win_size))+canny_edge(std_dev(b, win_size)), 
        0, 255
    )   

    show_imgs({
        "BW": IMG_BW,
        # "std x": std_dev(x, win_size),
        # "std y": std_dev(y, win_size),
        "grad bw": normalize(grad(IMG_BW)),
        # "LAB sc Lch": normalize(l_channel),
        # "threshold": smooth(fill_edge(TEST)-TEST),
        # "test edge overlap": TEST,
    })