import cv2
import numpy as np

from helper import DATA_PATH

# Paths to the model files
prototxt_path = "deploy.prototxt"
caffemodel_path = "hed_pretrained_bsds.caffemodel"

# Load the pre-trained HED model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Read the input image
# image_path = "data\Rohdaten\part_3\mask_20241126-154623-554.png"
image_path = DATA_PATH / "part_20/mask_20241202-114431-044.png"
image: np.ndarray = cv2.imread(image_path)

NET_BORDER = 50

image = cv2.copyMakeBorder(image, NET_BORDER, NET_BORDER, NET_BORDER, NET_BORDER, cv2.BORDER_CONSTANT)
# image = cv2.GaussianBlur(image, (3,3), 0)
image = cv2.bilateralFilter(image, 10, 30, 15)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


SCALE = image.shape[:2]
SCALE = SCALE[::-1]
# Resize the image to match the input size expected by the model
input_image = cv2.resize(image, (500, 500))
# Convert the image to a blob
r, g, b = cv2.split(image)
blob = cv2.dnn.blobFromImage(input_image, scalefactor=1, size=(500, 500),
                             mean=(r.mean(), g.mean(), b.mean()),
                             swapRB=True, crop=False)

net.setInput(blob)
edges = net.forward()

# The output is a single-channel image, scale it back to the original size
edges = edges[0, 0]
cv2.imshow("bin", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalize the edges for visualization (0-255 range)
edges = (255 * edges).astype("uint8")
# post-processing
from skimage.exposure import equalize_adapthist
# Display the edges
edges = edges[NET_BORDER:, NET_BORDER:]
# edges = cv2.GaussianBlur(edges, (3,3), 0)
edges = cv2.bilateralFilter(edges, 10, 60, 30)
edges = equalize_adapthist(edges)
cv2.imshow("eq", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# thresholding
from skimage.filters import threshold_otsu
thresh = threshold_otsu(edges)
edges = 255 * (edges > thresh).astype(np.uint8)
cv2.imshow("bin", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

from skimage.morphology import disk, dilation



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
    return filled_image - edge

flooded = fill_edge(edges)
flooded = dilation(flooded, disk(2))

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Find the bounding box of the largest contour
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
# Crop the image using the bounding box
flooded = flooded[y:y+h, x:x+w]
flooded = cv2.resize(flooded, SCALE)
# cv2.imshow("bin", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Visualize results
cv2.imshow("Original Binary Image", edges)
cv2.imshow("Flood-Filled Image", flooded)
cv2.imwrite("REF_PART20.png", flooded)
cv2.waitKey(0)
cv2.destroyAllWindows()