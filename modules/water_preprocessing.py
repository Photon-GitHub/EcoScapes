import os
from typing import override

import cv2
import numpy as np

from modules.module import Module, ModuleResult


class WaterPreprocessing(Module):
    def __init__(self):
        super().__init__("WaterPreprocessing", {"SatelliteLoader"})

    @override
    def main(self) -> ModuleResult:
        location = self.load_location()
        image_path = f"./satellite_data/{location}/water.png"

        # Read the image including the alpha channel for transparent (manually downloaded) images.
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Remove transparency if necessary.
        if len(img.shape) > 2 and img.shape[2] == 4:
            # Split the channels
            _, _, _, a = cv2.split(img)

            # Create a mask where alpha is zero (fully transparent)
            transparent_mask = (a == 0)

            # Convert to grayscale
            no_transparency_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            # Set fully transparent pixels to black
            no_transparency_img[transparent_mask] = 0
        else:
            no_transparency_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        _, binarized = cv2.threshold(no_transparency_img, 60, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        edited_image = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
        edited_image = cv2.morphologyEx(edited_image, cv2.MORPH_OPEN, kernel)
        edited_image = cv2.morphologyEx(edited_image, cv2.MORPH_DILATE, kernel)

        # Connected components analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edited_image, connectivity=8)

        # Create an output image to store the filtered components
        area_filtered = np.zeros_like(edited_image)

        for i in range(1, num_labels):  # Start from 1 to skip the background
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = float(width) / height if height > 0 else 0

            # Define your thresholds
            min_area = 100  # Adjust based on your specific needs
            min_aspect_ratio = 2  # Adjust to distinguish elongated shapes

            # Apply filters
            if area >= min_area or aspect_ratio >= min_aspect_ratio:
                area_filtered[labels == i] = 255  # Keep the component

        if not np.any(area_filtered == 255):
            return ModuleResult.STOP_PIPELINE

        color_output = cv2.cvtColor(area_filtered, cv2.COLOR_GRAY2BGR)

        dir_path = f"./satellite_image_processing/{location}"
        os.makedirs(dir_path, exist_ok=True)

        save_path = f"{dir_path}/water_preprocessed.png"
        cv2.imwrite(save_path, color_output)

        return ModuleResult.OK
