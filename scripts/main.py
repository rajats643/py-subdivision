import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import utils

from scripts.graph_based_image_segmentation import FHSolver, run_fh_rgb


def parse_config_file(config_file_path: Path) -> Dict:
    with open(config_file_path) as config_file:
        config: Dict = json.load(config_file)
    return config


def execute_workflow(
    sequence: int,
    blocks: Dict,
    image_paths: List[Path],
    output_path: Path,
    show_last: bool = False,
) -> bool:

    n: int = len(image_paths)
    for i, image_path in enumerate(image_paths):
        print(image_path)
        image: np.ndarray = utils.read_image(image_path)

        for block_id in range(sequence):
            block: Dict = blocks[str(block_id)]
            operation: str = block["operation"]

            if operation == "Scale":
                image = utils.scale_image(image, scale=block["scale_factor"])
            elif operation == "Gaussian":
                image = utils.gaussian_blur(
                    image, k=block["kernel_size"], sigma=block["sigma"]
                )
            elif operation == "Median":
                image = utils.median_blur(image, block["kernel_size"])
            elif operation == "FH":
                run_fh_rgb(image, k_factor=block["k_factor"])
            else:
                raise RuntimeError(f"Operation not supported: {operation}")

        if show_last and i == n - 1:
            utils.show_image(image)
        else:
            utils.write_image(
                image, (output_path / image_path.name).with_suffix(image_path.suffix)
            )

    return True


def extract_workflow(config: Dict) -> Tuple[int, Dict, List[Path], Path, bool]:
    """
    config contains all details required to run a workflow
    """

    # get output path
    output_path: Path = Path(config["output_path"]) if config["output_path"] else None

    # get temp image path
    temp_image_path: Path = (
        Path(config["data"]["image_path"]) if config["data"]["image_path"] else None
    )

    # get full image paths from directory
    full_image_list: List[Path] = []
    if config["data"]["directory_path"]:
        full_image_list = [
            Path(config["data"]["directory_path"]) / p
            for p in os.listdir(config["data"]["directory_path"])
        ]

    # verify data presence
    if full_image_list and (output_path is None):
        raise RuntimeError("No output path specified")
    if temp_image_path is None and not full_image_list:
        raise RuntimeError("No data provided")

    show_last: bool = False
    if temp_image_path:
        full_image_list.append(temp_image_path)
        show_last = True

    # values returned used in segmentation
    return (
        config["workflow"]["sequence"],
        config["workflow"]["blocks"],
        full_image_list,
        output_path,
        show_last,
    )


def main():
    """
    explain the project here
    """

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file-path",
        type=str,
        required=True,
        help="Path to the file containing workflow information",
    )

    # parse arguments and read config file
    args = parser.parse_args()
    config_file_path: Path = Path(args.config_file_path)
    config = parse_config_file(config_file_path)

    workflow: Tuple = extract_workflow(config)
    result = execute_workflow(*workflow)
    if not result:
        print("Execution failed")
    else:
        print("Execution successful")


if __name__ == "__main__":
    main()
