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
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    ret = normalize(clahe.apply(image))
    ret =  cv2.Canny(ret, 150, 150, L2gradient=True)
    kernel = np.ones((5,5))
    ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((3,3))
    # ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel)
    return normalize(ret)

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
        # "hsv_h": h, 
        "hsv_s": s, 
        "hsv_v": v, 
        # "rgb_r": r, 
        # "rgb_g": g, 
        # "rgb_b": b,
    }

    DATA = {}
    for key, c in channels.items():
        # DATA.append(c)
        DATA[f"{key}_grad"] = k_means_clust(grad(c))#.flatten()
        DATA[f"{key}_mean_dev"] = k_means_clust(canny_edge(abs_mean_dev(c)))#.flatten()
        DATA[f"{key}_std"] = k_means_clust(canny_edge(std_dev(c, win_size)))#.flatten()
        DATA[f"{key}_edge"] = k_means_clust(canny_edge(c))#.flatten()

    return DATA

def k_means_clust(image: np.ndarray, n: int = 2) -> np.ndarray:
    pixel_vals = image.flatten().astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    retval, labels, centers = cv2.kmeans(pixel_vals, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    return normalize(segmented_data.reshape((image.shape)))


if __name__ == "__main__":
    part_id = 30
    FILE_LIST = find_parts(DATA_PATH / f"part_{part_id}")
    NAME = FILE_LIST[1]
    print(f"LOADING: {NAME}")
    image = cv2.imread(NAME)
    b_size = 20
    image = cv2.copyMakeBorder(image, b_size, b_size, b_size, b_size, cv2.BORDER_CONSTANT)
    DATA = generate_feature_stack(image)
    show_imgs(DATA)

    import pandas as pd

    for key,feature in DATA.items():
        DATA[key] = feature.flatten()
    df = pd.DataFrame().from_dict(DATA)

    from sklearn.decomposition import PCA
    import seaborn as sb

    N_COMP = 10
    pca = PCA(n_components=N_COMP)

    pca.fit(df)

    pca_feature_weights = pd.DataFrame(
        pca.components_, columns=df.columns, index=[f"PC{ii}" for ii in range(N_COMP)] 
    )

    print(pca_feature_weights)

    pca_features = pd.DataFrame({
        col: df.to_numpy() @ pca_feature_weights.loc[col, :]
        for col in pca_feature_weights.index
    })

    # print(pca_features.info())
    print(sum(pca.explained_variance_ratio_))
    
    format = image.shape[:-1]
    PCA_DATA = {} 
    for key, val in pca_features.items():
        PCA_DATA[key] = normalize(val.to_numpy()).reshape(format)

    show_imgs(PCA_DATA)