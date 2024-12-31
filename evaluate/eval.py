from argparse import ArgumentParser
from pathlib import Path
from subprocess import check_call

import pandas as pd
import numpy as np


def main():
    # Parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--command",
        type=str,
        default="python ../solution/main.py",
        help="Command to run your program. See the README for the exact interface.",
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
    ground_truth = pd.read_csv(ground_truth_file)

    # Evaluate the results and print the output
    print()
    print("Evaluation result:")
    x_errors = np.abs(outputs["x"] - ground_truth["x"])
    print(
        f"\tAverage X Position Error (horizontal):\t {np.mean(x_errors):.2f} ±{np.std(x_errors):.2f} mm"
    )
    y_errors = np.abs(outputs["y"] - ground_truth["y"])
    print(
        f"\tAverage Y Position Error (vertical):\t {np.mean(y_errors):.2f} ±{np.std(y_errors):.2f} mm"
    )
    # The angle error is s bit mor tricky, as it is a circular value
    angle_errors = _difference_circular_range(
        outputs["angle"], ground_truth["angle"], 0.0, 360.0
    )
    print(
        f"\tAverage Angle Error:\t                 {np.mean(angle_errors):.2f} ±{np.std(angle_errors):.2f} degrees"
    )


def _difference_circular_range(
    value_a: np.ndarray, value_b: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    """Calculates differences on a circular number line, where minimum and maximum meet.

    Args:
        value_a: the first value
        value_b: the second value
        minimum: the minimum of the desired bounds
        maximum: the maximum of the desired bounds, assumed to be strictly larger than ``minimum``

    Returns:
        the normalized value in :math:`[0, (maximum - minimum)/2]`
    """

    span = maximum - minimum
    difference = (value_a - value_b) % span

    # take the smaller one of the two possible distances, i.e. the smaller path around the circular range
    return np.minimum(difference, span - difference)


if __name__ == "__main__":
    main()
