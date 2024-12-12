import cv2
import numpy as np
# from scipy.ndimage import gaussian_filter
import pandas as pd    
from sklearn.decomposition import PCA
    
# import the necessary packages
from skimage import feature
import numpy as np
    
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
    grid = None
    for idx, (label, img) in enumerate(images.items()):
        if grid is None:
            img_h, img_w = img.shape[:2]
            grid = np.zeros((grid_r * img.shape[0], grid_c * img.shape[1], 3), dtype=np.uint8)
        if len(img.shape) != 3:
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
    return normalize(np.abs(image - np.mean(image)))

def smooth(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5,5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # sigma = np.min(image.shape) / 100
    # return gaussian_filter(image, sigma)
    return image
    
def canny_edge(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5,5))
    image = normalize(clahe.apply(image))
    med_val = np.median(image) 
    # print(f"MED_VAL: {med_val}")
    lower = int(max(0 ,.7*med_val))
    upper = int(min(255,1.3*med_val))
    return cv2.Canny(image, lower, upper, L2gradient=True)
    
def morph_ex(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3,3))
    return  normalize(cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel))
    
def clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    return normalize(clahe.apply(image))

def apply_gabor_filter(image: np.ndarray, theta=0) -> tuple[np.ndarray]:
    """
    Apply a Gabor filter to the input image.
    
    Parameters:
    - image: Input grayscale image.
    - kernel_size: Size of the Gabor kernel.
    - sigma: Standard deviation of the Gaussian function.
    - theta: Orientation of the Gabor filter in radians.
    - lambd: Wavelength of the sinusoidal component.
    - gamma: Spatial aspect ratio (ellipticity).
    - psi: Phase offset.
    
    Returns:
    - filtered_image: Output image after applying the Gabor filter.
    """
    kernel_size = int(np.min(image.shape[:2]) / 20)
    sigma = np.min(image.shape[:2]) / 120
    lambd = np.min(image.shape[:2]) / 60
    gamma = np.min(image.shape[:2]) / 450
    psi = 0
    # print(kernel_size, sigma, lambd, gamma)
    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    
    return filtered_image, gabor_kernel

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

def generate_feature_stack(image: np.ndarray) -> np.ndarray:
    IMG_RGB = image.copy()
    IMG_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    IMG_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    l, x, y = cv2.split(IMG_LAB)
    h, s, v = cv2.split(IMG_HSV)
    r, g, b, = cv2.split(IMG_RGB)
    
    win_size = (np.array(bw.shape) / 50).astype(np.int16)
    channels = {
        "bw": bw, 
        "lab_l": l, 
        "lab_a": x, 
        "lab_b": y, 
        "hsv_h": h, 
        "hsv_s": s, 
        "hsv_v": v, 
        "rgb_r": r, 
        "rgb_g": g, 
        "rgb_b": b,
    }
    DATA = {}
    theta_values = [
        0, 
        np.pi/8, 
        np.pi/4, 
        3*np.pi/8, 
        np.pi/2, 
        5*np.pi/8, 
        3*np.pi/4, 
        7*np.pi/8,
    ]
    for key, c in channels.items():
        DATA[f"{key}_grad"] = grad(clahe(c))#.flatten()
        DATA[f"{key}_mean_dev_edge"] = canny_edge(abs_mean_dev(clahe(c)))#.flatten()
        DATA[f"{key}_std"] = abs_mean_dev(std_dev(c, win_size))#.flatten()
        DATA[f"{key}_edge"] = canny_edge(c)#.flatten()
        DATA[f"{key}"] = normalize(c)
        DATA[f"{key}_mean_dev"] = abs_mean_dev(c)
        DATA[f"{key}_laplacian"] = cv2.Laplacian(c, cv2.CV_64F)
        
        GAB = np.zeros_like(c,dtype=np.float64)
        for deg in theta_values:
            X,_ = apply_gabor_filter(clahe(c), theta=deg)
            X = k_means_clust(X, 4)
            X[X < np.max(X)] = 0
            GAB += normalize(X)
        DATA[f"{key}_gabor"] = clahe(normalize(GAB))  
        
        
        
        
    return DATA

def k_means_clust(image: np.ndarray, n: int = 2) -> np.ndarray:
    pixel_vals = image.flatten().astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    retval, labels, centers = cv2.kmeans(pixel_vals, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    return normalize(segmented_data.reshape((image.shape)))

def pca_transform(DATA: pd.DataFrame, N_COMP: int = 10) -> dict[str, np.ndarray]:
    pca = PCA(n_components=N_COMP)
    pca.fit(DATA)
    pca_feature_weights = pd.DataFrame(
        pca.components_, columns=DATA.columns, index=[f"PC{ii}" for ii in range(N_COMP)] 
    )
    # print(pca_feature_weights)
    pca_features = pd.DataFrame({
        col: DATA.to_numpy() @ pca_feature_weights.loc[col, :]
        for col in pca_feature_weights.index
    })
    # print(pca_features.info())
    print(f"PCA VARIANCE RATIO OF {N_COMP} Elements: {sum(pca.explained_variance_ratio_):.3f}/1")
    
    format = image.shape[:-1]
    PCA_DATA = {} 
    for key, val in pca_features.items():
        X = normalize(val.to_numpy().reshape(format))
        # print(f"{key} UNIQUE VAL: {np.unique(X)}")
        # X[X < np.max(X)] = 0
        PCA_DATA[key] = X

    return PCA_DATA

if __name__ == "__main__":
    part_id = 2
    FILE_LIST = find_parts(DATA_PATH / f"part_{part_id}")
    NAME = FILE_LIST[1]
    # NAME = "/home/user0/USERCODE/Hackathon-2024/data/Rohdaten/part_2/mask_20241203-164737-095.png"
    print(f"LOADING: {NAME}")
    image = cv2.imread(NAME)
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT)
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.bilateralFilter(image, 31, 75, 150)
    img_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # show_imgs({
    #     "filt": img_bw.copy(),
    #     "binary": normalize(feature.local_binary_pattern(img_bw, 8,1, method="nri_uniform")) 
    # })
    # image_hist = clahe(image_bw)
    # image_filt_filt = cv2.bilateralFilter(image_hist, 15, 75, 75)
    
    DATA = generate_feature_stack(image)
    # show_imgs(DATA)
    for key,feature in DATA.items():
        DATA[key] = feature.flatten()
    DATA = pd.DataFrame().from_dict(DATA)
    DATA = pca_transform(DATA, 3)        
    show_imgs(DATA)
