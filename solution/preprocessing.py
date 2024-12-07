import cv2
from pathlib import Path
from typing import Self
import numpy as np

import matplotlib.pyplot as plt
import mpld3
        
if __name__ == "__main__":
    from helper import DATA_PATH
else:
    from .helper import DATA_PATH

class Image:
    def __init__(self, file_path: Path) -> Self:
        self.file_path: Path = file_path
        self._img: np.ndarray = cv2.imread(filename=file_path)
    	
    def show(self) -> None:
        cv2.imshow(str(self.file_path), self._img)
        cv2.waitKey(1)
        input()

    def preprocessing_pipeline(self) -> None:
        IMG = self._img.copy()
        # self.__plot_rgb_ch_intensity_dist(IMG)
        # IMG = self.__apply_threshold(IMG)
        r_ch = IMG[:,:,1]
        b_ch = IMG[:,:,0]
        g_ch = IMG[:,:,2]
        img_bw = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        
        img_int_dist = Image.__rgb_ch_intensity_dist(img_bw)
        b_int_dist = Image.__rgb_ch_intensity_dist(b_ch)
        r_int_dist = Image.__rgb_ch_intensity_dist(r_ch)
        g_int_dist = Image.__rgb_ch_intensity_dist(g_ch)
        
        img_cum = Image.__rgb_channel_cumulative(img_int_dist)
        b_cum = Image.__rgb_channel_cumulative(b_int_dist)
        r_cum = Image.__rgb_channel_cumulative(r_int_dist)
        g_cum = Image.__rgb_channel_cumulative(g_int_dist)

        blue_thresh = len(b_cum[b_cum < 0.95])
        red_thresh = len(r_cum[r_cum < 0.95])
        green_thresh = len(g_cum[g_cum < 0.95])
        print(blue_thresh, red_thresh, green_thresh)

        r_ch[r_ch > red_thresh] = 0
        g_ch[g_ch > green_thresh] = 0
        b_ch[b_ch > blue_thresh] = 0
        self._img = IMG

    @staticmethod
    def __apply_threshold(img: np.ndarray) -> np.ndarray:
        img_int_dist, b_int_dist, r_int_dist, g_int_dist = Image.__rgb_ch_intensity_dist(img)
        
        img[img > 160] = 0
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bw = cv2.threshold(img_bw, 30, 255, cv2.THRESH_BINARY)
        # img[img_bw == 0] = 0 
        # img_bw_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_bw

    @staticmethod
    def __rgb_ch_intensity_dist(img_ch: np.ndarray, intensity_range: int = 1<<8) -> np.ndarray:
        ret_val,_ = np.histogram(img_ch, bins=np.arange(intensity_range))
        return ret_val
        
    @staticmethod
    def __rgb_channel_cumulative(img_ch: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.sum(img_ch[:ii+1]) for ii,_ in enumerate(img_ch)
        ]) / np.sum(img_ch)

    @staticmethod
    def __plot_rgb_channel_cumulative(img: np.ndarray) -> None:
        r_ch = img[:,:,1]
        b_ch = img[:,:,0]
        g_ch = img[:,:,2]
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img_int_dist = Image.__rgb_ch_intensity_dist(img_bw)
        b_int_dist = Image.__rgb_ch_intensity_dist(b_ch)
        r_int_dist = Image.__rgb_ch_intensity_dist(r_ch)
        g_int_dist = Image.__rgb_ch_intensity_dist(g_ch)
        
        img_cum = Image.__rgb_channel_cumulative(img_int_dist)
        b_cum = Image.__rgb_channel_cumulative(b_int_dist)
        r_cum = Image.__rgb_channel_cumulative(r_int_dist)
        g_cum = Image.__rgb_channel_cumulative(g_int_dist)

        plt.figure(figsize=(8,6))
        plt.plot(img_cum, label="bw image intensity cumulative", color="y")
        plt.plot(b_cum, label="blue channel intensity cumulative", color="b")
        plt.plot(r_cum, label="red channel intensity cumulative", color="r")
        plt.plot(g_cum, label="green channel intensity cumulative", color="g")
        plt.legend()
        plt.grid()
        plt.xlabel("channel value")
        plt.ylabel("N")
        mpld3.show()

    @staticmethod
    def __plot_rgb_ch_intensity_dist(img: np.ndarray) -> None:
        r_ch = img[:,:,1]
        b_ch = img[:,:,0]
        g_ch = img[:,:,2]
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_int_dist = Image.__rgb_ch_intensity_dist(img_bw)
        b_int_dist = Image.__rgb_ch_intensity_dist(b_ch)
        r_int_dist = Image.__rgb_ch_intensity_dist(r_ch)
        g_int_dist = Image.__rgb_ch_intensity_dist(g_ch)

        plt.figure(figsize=(8,6))
        plt.plot(img_int_dist, label="bw image intensity dist", color="y")
        plt.plot(r_int_dist, label="red channel intensity dist", color="r")
        plt.plot(b_int_dist, label="blue channel intensity dist", color="b")
        plt.plot(g_int_dist, label="green channel intensity dist", color="g")
        plt.legend()
        plt.grid()
        plt.xlabel("channel value")
        plt.ylabel("N")
        mpld3.show()

    @staticmethod
    def __plot_rgb_channels(img: np.ndarray) -> None:
        R = img.copy()
        R[:,:,[0,2]] = 0
        G = img.copy()
        G[:,:,[0,1]] = 0
        B = img.copy()
        B[:,:,[1,2]] = 0
        
        cv2.imshow("test1", R)
        cv2.imshow("test2", G)
        cv2.imshow("test3", B)
        cv2.imshow("img", img)
        cv2.waitKey(1)
        input()


if __name__ == "__main__":
    try:
        file_path: Path = DATA_PATH / "part_1/mask_20241202-170222-294.png"
        IMG = Image(file_path)
        IMG.preprocessing_pipeline()
        IMG.show()
    except Exception as E:
        print(E)
    finally:
        cv2.destroyAllWindows()