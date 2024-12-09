import cv2
from pathlib import Path
from typing import Self
import numpy as np

import matplotlib.pyplot as plt
import mpld3
        
if __name__ == "__main__":
    from helper import DATA_PATH, find_parts
else:
    from .helper import DATA_PATH, find_parts

class Image:
    def __init__(self, file_path: Path) -> Self:
        print(f"LOADING: {file_path}")
        self.file_path: Path = file_path
        self._img: np.ndarray = cv2.imread(filename=file_path)
    	
    def show(self) -> None:
        cv2.imshow(str(self.file_path), self._img)
        cv2.waitKey(1)
        input()

    def preprocessing_pipeline(self) -> None:
        b_size = 20
        self._img = cv2.copyMakeBorder(self._img, b_size, b_size, b_size, b_size, cv2.BORDER_CONSTANT)
        IMG = self._img.copy()
    
        from scipy.ndimage import gaussian_filter
        IMG = gaussian_filter(IMG, 1, 0)
        
        IMG_LAB = cv2.cvtColor(IMG, cv2.COLOR_BGR2LAB)
        IMG_HSV = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
        IMG_RGB = IMG.copy()
        l_channel, x, y = cv2.split(IMG_LAB)
        h, s, v = cv2.split(IMG_HSV)
        r, g, b, = cv2.split(IMG_RGB)
        r = self.__get_grad_norm_changes(r)
        g = self.__get_grad_norm_changes(g)
        b = self.__get_grad_norm_changes(b)
        x = self.__get_grad_norm_changes(x)
        y = self.__get_grad_norm_changes(y)
        l_channel = self.__get_grad_norm_changes(l_channel)
        hue = self.__get_grad_norm_changes(h)
        s = self.__get_grad_norm_changes(s)
        v = self.__get_grad_norm_changes(v)

        IMG:np.ndarray = l_channel+v#+s+h
        # IMG = gaussian_filter(IMG, 1, 0)
        IMG = 255*IMG / np.max(IMG)
        IMG = IMG.astype(np.uint8, casting="unsafe")
        
        IMG = cv2.Canny(IMG, 10, 255, L2gradient=True)
        IMG = gaussian_filter(IMG, .2, 0)
        
        kernel = np.ones((5,5))
        IMG = cv2.morphologyEx(IMG, cv2.MORPH_CLOSE, kernel)
        
        IMG = np.clip(IMG*10, 0, 255)
        _, IMG = cv2.threshold(IMG, 200, 255, cv2.THRESH_BINARY)        
        
        contours, _ = cv2.findContours(IMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        h, w = IMG.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)  # Add a 1-pixel border around the image

        # Seed point inside the largest contour
        # (You can use cv2.boundingRect to find a seed point)
        x, y, w, h = cv2.boundingRect(largest_contour)
        seed_point = (x + w // 2, y + h // 2)

        # Flood-fill the largest contour
        filled_image = IMG.copy()  # Make a copy of the binary image
        cv2.floodFill(filled_image, mask, seed_point, 255)
        filled_image -= IMG
        filled_image = cv2.cvtColor(filled_image, cv2.COLOR_GRAY2BGR)
        IMG = cv2.cvtColor(IMG, cv2.COLOR_GRAY2BGR)
        result = np.hstack((r, g, b, l_channel, hue, s, v))
        cv2.imshow("R-G-B-Lch-Hue-Sat-Val", result)
        cv2.imshow("image", self._img)
        cv2.waitKey(1)
        input()
        
        self._img = IMG
        

    @staticmethod
    def __get_grad_norm_changes(img_1d: np.ndarray) -> np.ndarray:
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(32,32))
        # img_1d = clahe.apply(img_1d)
        
        img_1d = np.gradient(img_1d)
        img_1d = np.linalg.norm(img_1d, axis=0)
        img_1d = np.array(img_1d/ np.max(img_1d))
        
        kernel = np.ones((3,3))
        img_1d = cv2.morphologyEx(img_1d, cv2.MORPH_CLOSE, kernel)
        return img_1d

if __name__ == "__main__":
    try:
        part_id =13
        FILE_LIST = find_parts(DATA_PATH / f"part_{part_id}")
        IMG = Image(FILE_LIST[3])
        IMG.preprocessing_pipeline()
        # IMG.show()
    except Exception as E:
        raise E
    finally:
        cv2.destroyAllWindows()


# IMG = enhanced_img
        # img_bw = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
        # img_bw[img_bw == 0] = 1
        # max_int = 255 * np.ones_like(img_bw)

        # compression_mask = np.divide(max_int, img_bw)
        # compression_mask[compression_mask == np.inf] = 0
        # for ch in range(IMG.shape[2]):
        #     IMG[:,:,ch] = np.multiply(IMG[:,:,ch], compression_mask)
        
        # img_bw_compressed = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        # img_int_dist = Image.__rgb_ch_intensity_dist(img_bw_compressed)
        
        # from scipy.signal import find_peaks, peak_prominences
        # peaks,_ = find_peaks(img_int_dist, width=5, rel_height=10)

        # prominences, l, r = peak_prominences(img_int_dist, peaks) 
        # h_peak = np.argmax(prominences)

        # img_bw_compressed[img_bw_compressed < l[h_peak]] = 0
        # img_bw_compressed[img_bw_compressed > r[h_peak]] = 0
        # img_bw_compressed[img_bw_compressed != 0] = 255
        # print()
        
        # # self.__plot_rgb_channel_cumulative(IMG)
        # # self.__plot_rgb_ch_intensity_dist(IMG)
        # self._img = IMG

    # @staticmethod
    # def __rgb_ch_intensity_dist(img_ch: np.ndarray, intensity_range: int = 1<<8) -> np.ndarray:
    #     ret_val,_ = np.histogram(img_ch, bins=np.arange(intensity_range))
    #     return ret_val
        
    # @staticmethod
    # def __rgb_channel_cumulative(img_ch: np.ndarray) -> np.ndarray:
    #     return np.asarray([
    #         np.sum(img_ch[:ii+1]) for ii,_ in enumerate(img_ch)
    #     ]) / np.sum(img_ch)

    # @staticmethod
    # def __plot_rgb_channel_cumulative(img: np.ndarray) -> None:
    #     r_ch = img[:,:,1]
    #     b_ch = img[:,:,0]
    #     g_ch = img[:,:,2]
    #     img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #     img_int_dist = Image.__rgb_ch_intensity_dist(img_bw)
    #     b_int_dist = Image.__rgb_ch_intensity_dist(b_ch)
    #     r_int_dist = Image.__rgb_ch_intensity_dist(r_ch)
    #     g_int_dist = Image.__rgb_ch_intensity_dist(g_ch)
        
    #     img_cum = Image.__rgb_channel_cumulative(img_int_dist)
    #     b_cum = Image.__rgb_channel_cumulative(b_int_dist)
    #     r_cum = Image.__rgb_channel_cumulative(r_int_dist)
    #     g_cum = Image.__rgb_channel_cumulative(g_int_dist)

    #     plt.figure(figsize=(8,6))
    #     plt.plot(img_cum, label="bw image intensity cumulative", color="y")
    #     plt.plot(b_cum, label="blue channel intensity cumulative", color="b")
    #     plt.plot(r_cum, label="red channel intensity cumulative", color="r")
    #     plt.plot(g_cum, label="green channel intensity cumulative", color="g")
    #     plt.legend()
    #     plt.grid()
    #     plt.xlabel("channel value")
    #     plt.ylabel("N")
    #     mpld3.show()

    # @staticmethod
    # def __plot_rgb_ch_intensity_dist(img: np.ndarray) -> None:
    #     r_ch = img[:,:,1]
    #     b_ch = img[:,:,0]
    #     g_ch = img[:,:,2]
    #     img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img_int_dist = Image.__rgb_ch_intensity_dist(img_bw)
    #     b_int_dist = Image.__rgb_ch_intensity_dist(b_ch)
    #     r_int_dist = Image.__rgb_ch_intensity_dist(r_ch)
    #     g_int_dist = Image.__rgb_ch_intensity_dist(g_ch)

    #     plt.figure(figsize=(8,6))
    #     plt.plot(img_int_dist, label="bw image intensity dist", color="y")
    #     plt.plot(r_int_dist, label="red channel intensity dist", color="r")
    #     plt.plot(b_int_dist, label="blue channel intensity dist", color="b")
    #     plt.plot(g_int_dist, label="green channel intensity dist", color="g")
    #     plt.legend()
    #     plt.grid()
    #     plt.xlabel("channel value")
    #     plt.ylabel("N")
    #     mpld3.show()