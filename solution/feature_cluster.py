import cv2
import numpy as np
import pandas as pd    
from sklearn.decomposition import PCA    
import skimage as ski
from skimage import filters
from skimage import feature
from skimage.morphology import disk

if __name__ == "__main__":
    from helper import DATA_PATH, find_parts
else:
    from .helper import DATA_PATH, find_parts

class Features:
    @staticmethod
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

    @staticmethod
    def show_imgs(images: dict[str, np.ndarray]) -> None:
        n_img = len(images)
        grid_r, grid_c = Features.__optimal_grid(n_img)
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
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    	
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        return np.asarray(255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))).astype(np.uint8)

    @staticmethod
    def std_dev(image: np.ndarray, win_size: tuple[int]) -> np.ndarray:
        mean_loc_sq = cv2.filter2D(image, -1, np.ones(win_size))
        return Features.normalize(np.sqrt((image-mean_loc_sq)**2))

    @staticmethod
    def abs_mean_dev(image: np.ndarray) -> np.ndarray:
        return Features.normalize(np.abs(image - np.mean(image)))

def k_means_clust(image: np.ndarray, n: int = 2) -> np.ndarray:
    pixel_vals = Features.normalize(image).flatten().astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    retval, labels, centers = cv2.kmeans(pixel_vals, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    return Features.normalize(segmented_data.reshape((image.shape)))

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
        "bw": Features.normalize(bw), 
        "lab_l": Features.normalize(l), 
        "lab_color": Features.normalize(x+y),
        "hsv_color": Features.normalize(h+s),
        # "lab_a": x, 
        # "lab_b": y, 
        # "hsv_h": h, 
        # "hsv_s": s, 
        # "hsv_v": v, 
        # "rgb_r": r, 
        # "rgb_g": g, 
        # "rgb_b": b,
    }
    DATA = {}
    for key, c in channels.items():
        butter = Features.normalize(filters.butterworth(c, 0.01))
        img = butter.copy()
        # img[img == 0] = img.mean()
        img = img_as_float(img)
        # img = gaussian_filter(img, 1)
        
        seed_er = img.copy()
        seed_er[1:-1, 1:-1] = img.max()
        mask_er = img

        seed_dil = img.copy()
        seed_dil[1:-1, 1:-1] = img.min()
        mask_dil = img
        from skimage.exposure import equalize_adapthist

        dilated = reconstruction(seed_dil, mask_dil, method="dilation")
        eroded = reconstruction(seed_er, mask_er, method="erosion")
        DATA.update({
            # f"{key}": c,
            # f"{key}_std": abs_mean_dev(std_dev(c, win_size)),
            # f"{key}_laplacian": cv2.Lap#lacian(c, cv2.CV_64F),
            # f"{key}_binary": Features.normalize(
            #     ski.feature.local_binary_pattern(c, 8,1, method="uniform")
            # ),
            # f"{key}_clahe": Features.normalize(
            #     equalize_adapthist(c)
            # ),
            f"{key}_butter": butter,
            # f"{key}_eroded": Features.normalize(eroded), 
            # f"{key}_dilated": Features.normalize(dilated), 
            f"{key}_farid": Features.normalize(
                filters.farid(c)
            ),
            # f"{key}_roberts": Features.normalize(filters.roberts(img_bw.copy())),
            # f"{key}_scharr": Features.normalize(
            #     filters.scharr(c.copy())
            # ),
            f"{key}_canny": Features.normalize(
                feature.canny(butter).astype(np.uint8)
            ),
            # f"{key}_entropy": Features.normalize(
            #     filters.rank.entropy(c.copy(), disk(3))
            # ),
        })
    return DATA

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
        X = Features.normalize(val.to_numpy().reshape(format))
        # print(f"{key} UNIQUE VAL: {np.unique(X)}")
        # X[X < np.max(X)] = 0
        PCA_DATA[key] = k_means_clust(X, 3)

    return PCA_DATA

if __name__ == "__main__":
    part_id = 1
    FILE_LIST = find_parts(DATA_PATH / f"part_{part_id}")
    NAME = FILE_LIST[1]
    # NAME = "/home/user0/USERCODE/Hackathon-2024/data/Rohdaten/part_2/mask_20241203-164737-095.png"
    print(f"LOADING: {NAME}")
    image = cv2.imread(NAME)
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT)
    # image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.bilateralFilter(image, 15, 75, 30)
    img_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_float
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import reconstruction


    # img = img_bw.copy()
    # img[img == 0] = img.mean()
    # img = img_as_float(img)
    # # img = gaussian_filter(img, 1)
    
    # seed = np.copy(img)
    # seed[1:-1, 1:-1] = img.min()
    # mask = img

    # dilated = reconstruction(seed, mask, method="dilation")
    # DATA = {
    #     "bw": img_bw,
    #     "test": Features.normalize(dilated)
    # }
    DATA = generate_feature_stack(image)
    Features.show_imgs(DATA)
    # for key,feature in DATA.items():
    #     DATA[key] = feature.flatten()
    # DATA = pd.DataFrame().from_dict(DATA)
    # DATA = pca_transform(DATA, 2)        
    # Features.show_imgs(DATA) 


    # from skimage.morphology import diameter_closing, diameter_opening, dilation, area_opening
    # for key, pc in DATA.items():
    #     X = Features.normalize(area_opening(pc, 4, 2))
    #     X = Features.normalize(dilation(X, disk(3)).astype(int))
    #     DATA[key] = X
    # Features.show_imgs(DATA)

    