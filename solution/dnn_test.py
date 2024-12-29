import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu


from pathlib import Path
from typing import Self
from cv2.dnn import Net

from helper import DATA_PATH

def show(image: np.ndarray, title: str = "Image", scale: float = 1.) -> None:
    imsize = image.shape[:2]
    imsize: tuple = imsize[::-1]
    image = cv2.resize(
        image, 
        (imsize[0]*scale, imsize[1]*scale)
    )
    cv2.imshow(title, image)
    cv2.waitKey(0)

def align_to_boundary(image: np.ndarray) -> tuple[int, int, np.ndarray]:
    thresh = threshold_otsu(image)
    binary = 255 * (image > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Find the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Crop the image using the bounding box
    image = image[y:y+h, x:x+w]
    return x, y, image

def normalize(image: np.ndarray) -> np.ndarray:
        return np.asarray(255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))).astype(np.uint8)

class DNNEdge(object):
    NET_IMG_SHAPE: tuple[int, int] = (500, 500)
    NET_BORDER: int = 50

    def __init__(
        self, 
        prototxt_path: Path|str = "deploy.prototxt",
        caffemodel_path: Path|str = "hed_pretrained_bsds.caffemodel",
    ) -> Self:
        # Load the pre-trained HED model
        self.net: Net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def get_edges(self, image: np.ndarray) -> np.ndarray:
        SCALE = image.shape[:2]
        SCALE = SCALE[::-1]
        
        image = self.__image_preprocessing(image)
        blob = self.__img_to_blob(image)
        self.net.setInput(blob)
        edges = self.net.forward()[0, 0]
        edges = self.__edge_postprocessing(edges)
        _, _, edges = align_to_boundary(edges)
        edges = cv2.resize(edges, SCALE)
        return edges

    @staticmethod
    def __image_preprocessing(image: np.ndarray) -> np.ndarray:
        image = cv2.copyMakeBorder(
            image, 
            DNNEdge.NET_BORDER, 
            DNNEdge.NET_BORDER, 
            DNNEdge.NET_BORDER, 
            DNNEdge.NET_BORDER, 
            cv2.BORDER_CONSTANT
        )
        # image = cv2.GaussianBlur(image, (3,3), 0)
        image = cv2.bilateralFilter(image, 10, 30, 15)
        return image

    @staticmethod
    def __img_to_blob(image: np.ndarray) -> np.ndarray:
        # Resize the image to match the input size expected by the model
        input_image = cv2.resize(image, DNNEdge.NET_IMG_SHAPE)
        # Convert the image to a blob
        r, g, b = cv2.split(image)
        return cv2.dnn.blobFromImage(
            input_image, 
            scalefactor=1, 
            size=DNNEdge.NET_IMG_SHAPE,
            mean=(r.mean(), g.mean(), b.mean()),
            swapRB=True, 
            crop=False
        )

    @staticmethod
    def __edge_postprocessing(edges: np.ndarray) -> np.ndarray:
        edges = edges[DNNEdge.NET_BORDER:, DNNEdge.NET_BORDER:]
        # Normalize the edges for visualization (0-255 range)
        edges = (255 * edges).astype("uint8")
        edges = equalize_adapthist(edges)
        
        thresh = threshold_otsu(edges)
        edges = 255 * (edges > thresh).astype(np.uint8)
        edges = cv2.bilateralFilter(edges, 10, 60, 30)
        return edges


if __name__	== "__main__":
    # image_path = DATA_PATH / "part_3/ mask_20241126-154623-554.png"  # Ganz i.O.
    image_path = DATA_PATH / "part_20/mask_20241202-114431-044.png"   # Schrift ausgestanzt
    # image_path = DATA_PATH / "part_22/mask_20241203-165823-809.png"   # Riesen Loch in der Mitte
    # image_path = DATA_PATH / "part_1/mask_20241203-084242-404.png"    # Beschissener Hintergrund
    image: np.ndarray = cv2.imread(image_path)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_bw = equalize_adapthist(image_bw) 
    thresh = threshold_otsu(image_bw)
    flooding_mask = 255 * (image_bw > thresh).astype(np.uint8)        
    show(flooding_mask, "flooding_mask small features (holes)", 5)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = equalize_adapthist(image_hsv)
    from skimage import feature
    edges_hue = 255*feature.canny(image_hsv[:,:,0]).astype(np.uint8)
    from skimage.morphology import disk, closing
    edges_hue = closing(edges_hue, disk(3))
    # show(edges_hue, "thresh", 5)

    Edge = DNNEdge()
    edges = Edge.get_edges(image)
    # edges = closing(edges, disk(3))
    # show(edges, "Edges", 5)

    edges_total = normalize(edges + edges_hue)
    edges_total = closing(edges_total, disk(2))
    
    show(edges_total, "EDGES TOTAL", 5)


# exit()
    

    from skimage.morphology import disk, dilation



    def fill_edge_profile(edge: np.ndarray) -> np.ndarray:
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
        return filled_image - edge

    flooded = fill_edge_profile(edges_total)
    flooded = closing(flooded, disk(2))
    flooded[flooding_mask == 0] = 0
    show(flooded, "flooded", 5)
    