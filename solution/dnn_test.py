import cv2
import numpy as np

# Paths to the model files
prototxt_path = "deploy.prototxt"
caffemodel_path = "hed_pretrained_bsds.caffemodel"

# Load the pre-trained HED model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Read the input image
image_path = "data\Rohdaten\part_3\mask_20241126-154623-554.png"
image = cv2.imread(image_path)

# Resize the image to match the input size expected by the model
input_image = cv2.resize(image, (500, 500))
(h, w) = input_image.shape[:2]

# Convert the image to a blob
blob = cv2.dnn.blobFromImage(input_image, scalefactor=1.0, size=(500, 500),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)

# Perform forward pass to get the edge detection output
net.setInput(blob)
edges = net.forward()

# The output is a single-channel image, scale it back to the original size
edges = edges[0, 0]
edges = cv2.resize(edges, (w, h))

# Normalize the edges for visualization (0-255 range)
edges = (255 * edges).astype("uint8")

# Display the edges
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()