import cv2
import numpy as np
import random

def vibe_edge_detection(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found.")
        return

    # Step 1: Add some chaos - Randomized blurring
    kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])  # Vibe says pick random
    blurred = cv2.GaussianBlur(image, kernel_size, 0)

    # Step 2: Random thresholds for vibes
    low_thresh = random.randint(50, 100)
    high_thresh = random.randint(150, 200)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # Step 3: Combine with the OG image - spice it up
    combined = cv2.bitwise_or(image, edges)

    # Step 4: Enhance vibes with colormap
    colored_edges = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    # Step 5: Add some noise, why not
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_combined = cv2.add(combined, noise)

    # Step 6: Blend for *chef's kiss*
    final_vibe = cv2.addWeighted(noisy_combined, 0.7, edges, 0.3, 0)

    # Display the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Vibes Edge Detection", final_vibe)
    cv2.imshow("Colored Edges", colored_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Path to your image
# vibe_edge_detection("/home/user0/usercode/Hackathon-2024/data/Rohdaten/part_13/mask_20241202-164904-664.png")
# vibe_edge_detection("/home/user0/usercode/Hackathon-2024/data/Rohdaten/part_8/mask_20241202-165639-863.png")
# vibe_edge_detection("/home/user0/usercode/Hackathon-2024/data/Rohdaten/part_6/mask_20241203-165454-618.png")
# vibe_edge_detection("/home/user0/usercode/Hackathon-2024/data/Rohdaten/part_1/mask_20241203-084242-404.png")
# vibe_edge_detection("/home/user0/usercode/Hackathon-2024/data/Rohdaten/part_3/mask_20241126-154116-082.png")
image = cv2.imread("/home/user0/usercode/Hackathon-2024/data/Rohdaten/part_1/mask_20241203-084242-404.png")

from skimage.filters import difference_of_gaussians, window

# wimage = image * window('hann', image.shape)

filtered = difference_of_gaussians(image, .1, 10)
# wfiltered = filtered * window('hann', image.shape)

cv2.imshow("Windowed Image", image)
cv2.imshow("Windowed Filtered Image", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
