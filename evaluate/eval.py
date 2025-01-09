from argparse import ArgumentParser
from pathlib import Path
from subprocess import check_call

import pandas as pd
import numpy as np
import cvzone
import cv2
import numpy as np


def overlay_gripper_on_part(gripper_image, part_image, x, y, angle):
    """
    Overlay the gripper image onto the part image with specified translation and rotation.

    :param gripper_image: Input image of the gripper.
    :param part_image: Input image of the part mask.
    :param x: Horizontal offset (pixels) for gripper placement on the part.
    :param y: Vertical offset (pixels) for gripper placement on the part.
    :param angle: Rotation angle of the gripper in degrees.
    :return: Combined image with the gripper overlayed on the part., Boolean flag indicating if the gripper intersects with background.
    """
    # Rotate the gripper image as first transformation step
    rotated_gripper = cvzone.rotateImage(
        gripper_image, angle=angle, scale=1, keepSize=False
    )

    # Calculate offsets for translation step
    # Dimensions of the gripper and part images
    gh, gw = rotated_gripper.shape[:2]

    # Pad the part image so that the gripper can be out of bounds, original is centered
    ph, pw = part_image.shape[:2]
    pad_top, pad_bottom = gh, gh
    pad_left, pad_right = gw, gw
    part_image_padded = (
        np.ones(
            (ph + (pad_top + pad_bottom), pw + (pad_left + pad_right), 3),
            dtype=np.uint8,
        )
        * 255
    )
    part_image_padded[
        (pad_top) : (pad_top + ph), (pad_left) : (pad_left + pw)
    ] = part_image

    # apply tranformation offset
    overlay_x = int(x - (gw / 2) + pad_left)
    overlay_y = int(y - (gh / 2) + pad_top)

    # Only for visualization
    part_image_copy = part_image_padded.copy()
    mask_combined = cvzone.overlayPNG(
        part_image_copy, rotated_gripper, [overlay_x, overlay_y]
    )

    ## detect intersections

    # make sure mask is 0,1 for instersection
    part_hsv = cv2.cvtColor(part_image_padded, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 20])
    upper_white = np.array([180, 255, 255])  # HSV (0-180, 0-255, 0-255)

    part_hole_mask = cv2.inRange(part_hsv, lower_white, upper_white)

    # Create a mask for the non-transparent regions of the gripper ,e.g the suction points
    alpha_channel = rotated_gripper[:, :, 3]  # Extract the alpha channel
    gripper_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[
        1
    ]  # Binary mask for visible regions

    # Align the visible mask with the part's coordinate system
    aligned_gripper_mask = np.zeros(part_image_padded.shape[:2], dtype=np.uint8)
    aligned_gripper_mask[
        overlay_y : overlay_y + gh, overlay_x : overlay_x + gw
    ] = gripper_mask

    # Check for intersection
    intersection = cv2.bitwise_and(aligned_gripper_mask, part_hole_mask)
    intersection_exists = np.any(intersection)

    if intersection_exists:
        print("Intersection detected!")

    return mask_combined, intersection_exists


def calculate_score(part_image, x, y, intersection=False):
    """
    Calculate the score of the gripper placement on the part.

    :param part_image: Input image of the part (background).
    :param x: Horizontal offset (pixels) for gripper placement on the part.
    :param y: Vertical offset (pixels) for gripper placement on the part.
    :param intersection: Boolean flag indicating if the gripper intersects with holes.
    :return: Score of the gripper placement on the part. In Percent of part diagonal.
    """

    # get part image shape
    ph, pw = part_image.shape[:2]

    if intersection:
        # Constraint nicht erfüllt. maximale Distanz punkte!
        return 1

    ph, pw = part_image.shape[:2]
    diag = np.sqrt(ph**2 + pw**2)

    # Calculate the score
    return np.sqrt((x - (pw / 2)) ** 2 + (y - (ph / 2)) ** 2) / diag


def main():
    # Parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--command",
        type=str,
        default="python solution/main.py",
        help="Command to run your program. See the README for the exact interface.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, which displays the overlay images.",
    )
    args = parser.parse_args()
    command = args.command.strip()
    if not command:
        raise ValueError("Command is empty")

    # Paths
    directory = Path(__file__).parent
    input_file = directory / "task.csv"
    output_file = directory / "tool_output.csv"
    ground_truth_file = directory / "ground_truth.csv"

    # Call the program
    return_code = check_call([*command.split(" "), str(input_file), str(output_file)])
    if return_code != 0:
        raise ValueError("The program did not run successfully")
    if not output_file.exists():
        raise ValueError("Output file does not exist")

    # Read the results
    outputs = pd.read_csv(output_file)
    ground_truth_list = pd.read_csv(ground_truth_file)

    # iterate over the lines of output and ground truth
    for index, output_row in outputs.iterrows():
        ground_truth_row = ground_truth_list.iloc[index]
        output_part = Path(output_row["part"])
        output_gripper = Path(output_row["gripper"])
        ground_truth_part = Path(ground_truth_row["part"])
        ground_truth_gripper = Path(ground_truth_row["gripper"])
        if not (
            output_part == ground_truth_part and output_gripper == ground_truth_gripper
        ):
            print("Output and ground truth do not match!")
            print("Output: ----------")
            print(output_row)
            print("Ground truth: ----------")
            print(ground_truth_row)
            continue  ## end this iteration here

        # Part matches!

        # Check constraint
        # load mask image from ground truth
        mask_image = cv2.imread(Path(ground_truth_row["mask"]), cv2.IMREAD_UNCHANGED)
        # load gripper image from ground truth
        gripper_image = cv2.imread(
            Path(ground_truth_row["gripper"]), cv2.IMREAD_UNCHANGED
        )

        if len(mask_image.shape) < 3:
            # convert to 3 channel image
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

        # Extract x, y, and angle from output_row
        x_output = int(output_row["x"])
        y_output = int(output_row["y"])
        angle_output = -int(output_row["angle"])
        
        # Overlay gripper on part
        res, intersect = overlay_gripper_on_part(
            gripper_image=gripper_image,
            part_image=mask_image,
            x=x_output,
            y=y_output,
            angle=angle_output,
        )

        # Display the resulting overlay, you may comment this out.
        if args.debug:
            cv2.imshow("Overlay part and gripper", res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        outputs.loc[index, "score"] = calculate_score(
            mask_image, x_output, y_output, intersection=intersect
        )

    # Write the updated DataFrame back to the output file
    outputs.to_csv(output_file, index=False)

    print(f'Average score: {outputs["score"].mean()}')

    # Write the updated DataFrame back to the output file
    outputs.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
