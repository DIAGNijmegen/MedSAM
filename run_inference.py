from typing import Tuple, Any, List
from loguru import logger
import SimpleITK as sitk
from MedSAM_Inference import medsam_inference
from segment_anything import sam_model_registry
from skimage import transform
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import argparse

# CSV columns headers
COLUMNS = ['SeriesInstanceUID', 'CoordX', 'CoordY', 'CoordZ', 'Diameter [mm]', "Segmentation filename", "Slice Number"]


def read_file(file_path: str) -> Tuple[sitk.Image, np.ndarray]:
    """
    Read scan and return scan in simpleitk format and numpy array.
    Args:
        file_path: path to scan file

    Returns:
        image in simpleitk and numpy arrray format
    """
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    return image, image_array


def convert_nodule_annotation_to_bbox(coord_x: float, coord_y: float, coord_z: float, diameter: float,
                                      ref_img: sitk.Image) -> Tuple[Any]:
    """
    Convert a nodule annotation in real world coordinates to bounding box format [x1, y1, z1, width, height, depth]
    Diameter needs to be in mm. Reference image is used for spacing, origin, etc.
    Args:
        coord_x: x coordinate
        coord_y: y coordinate
        coord_z: z coordinate
        diameter: nodule diameter in mm
        ref_img: reference image

    Returns: nodule annotation in bounding box format [x1, y1, z1, width, height, depth]
    """
    nodule_coord = ref_img.TransformPhysicalPointToIndex((coord_x, coord_y, coord_z))
    diameter_px = diameter / ref_img.GetSpacing()[0]
    bbox = (nodule_coord[0] - int(round(diameter_px / 2)),
            nodule_coord[1] - int(round(diameter_px / 2)),
            nodule_coord[2] - int(round(diameter_px / 2)),
            int(diameter_px), int(diameter_px), int(diameter_px))
    return bbox


def extract_image_and_coordinates(csv_row: Tuple[int, Any], image: sitk.Image, scan_name: str) -> Tuple[List[int], int]:
    """
    Extract information (bbox, slice index) from csv file
    Args:
        csv_row: csv row from csv file
        image: reference image
        scan_name: scan name

    Returns: bounding box csv slice index number containing the nodule
    """
    try:
        coord_x, coord_y, coord_z, diameter = csv_row[1]["CoordX"], csv_row[1]["CoordY"], csv_row[1]["CoordZ"], \
            csv_row[1]["Diameter [mm]"]
        csv_slice_idx = int(csv_row[1]["Slice Number"])
        nodule_bbox = convert_nodule_annotation_to_bbox(coord_x=coord_x, coord_y=coord_y,
                                                        coord_z=coord_z,
                                                        diameter=diameter, ref_img=image)
        x, y, z, w, h = nodule_bbox[0], nodule_bbox[1], nodule_bbox[2], nodule_bbox[3], nodule_bbox[4]
        y2 = y + h
        x2 = x + w
        bbox = [x, y, x2, y2]

        return bbox, csv_slice_idx
    except:
        logger.error(f"Error in: {scan_name}")


def get_mm_box(centre_pixel: List[int], bbox_size_in_mm: float, pixel_spacing: np.ndarray) -> Tuple[Any]:
    """
    Get the bounding box of specified mm around the nodule
    Args:
        centre_pixel: center pixel of the nodule
        bbox_size_in_mm: bounding box size in mm
        pixel_spacing: pixel spacing

    Returns: x and y coordinates for the desired bounding box of x mm in size
    """
    box_size_pixels = bbox_size_in_mm / pixel_spacing
    box_size_left = np.round(box_size_pixels / 2).astype(int)
    box_size_right = np.round(box_size_pixels - box_size_left).astype(int)
    starts = centre_pixel - box_size_left
    ends = centre_pixel + box_size_right
    x1, y1, x2, y2 = starts[0], starts[1], ends[0], ends[1]
    return x1, y1, x2, y2


def get_n_slices(z_in_mm: float, pixel_spacing_z: float) -> int:
    """
    Calculate the number of slices in the z direction for a specified distance in mm
    Args:
        z_in_mm: z in mm
        pixel_spacing_z: pixel spacing for the z axis

    Returns: number of slices

    """
    n_slices = int(z_in_mm / pixel_spacing_z)
    return n_slices


def medsam_segmentation(model_path: str, image: np.ndarray, bbox: List[int], device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inference using MedSAM model.
    Args:
        model_path: path to MedSAM model weights file
        image: input image
        bbox: bounding box where the nodule is located (used as a prompt for inference)
        device: specific device to run inference on (cpu, cuda or mps)

    Returns: MedSAM inferenced mask
    """
    medsam_model = sam_model_registry["vit_b"](checkpoint=model_path).to(device)
    medsam_model.eval()
    img_np = image

    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    # Image preprocessing
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # Normalize to [0, 1], (H, W, 3)
    # Convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    box_np = np.array([bbox])
    # Transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

    return medsam_seg, box_np


def run_inference(input_path: str, csv_path: str,  output_path: str, model_path: str, device: str = "cpu",
                  bbox_mm: int = None) -> None:
    """
    Run inference iterating through every scan in the specified csv file and save the results to the output folder.
    Args:
        input_path: path to the folder containing the scans
        csv_path: path to the csv file containing nodule information
        output_path: path to the output folder to save the results
        model_path: path to the model weights file
        device: device to run inference on (cpu, cuda or mps)
        bbox_mm: size of bounding box (in mm) around the nodule, if not specified it will use nodule diameter + 5 pixels
    """
    csv_file = pd.read_csv(csv_path, usecols=COLUMNS, sep=";")
    for csv_row in csv_file.iterrows():
        file_path = Path(input_path) / str(csv_row[1]["SeriesInstanceUID"] + ".mha")
        processed_slices = []
        logger.info(f"Processing: {file_path}.")
        scan_name = str(file_path.parent).split("/")[-1]
        image, image_array = read_file(file_path=str(file_path))
        bbox, csv_slice_idx = extract_image_and_coordinates(csv_row=csv_row, image=image,
                                                                                  scan_name=scan_name)
        csv_slice_idx = csv_slice_idx - 1  # Because CT slices start from 1 and arrays start from 0
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w, h = x2 - x1, y2 - y1

        if bbox_mm is None:
            # Make the bbox from annotations 5 pixels bigger from each coordinate to improve MEDSAM segmentation
            x1, y1, x2, y2 = x1 - 5, y1 - 5, x2 + 5, y2 + 5
            nodule_diameter = csv_row[1]["Diameter [mm]"] + 5

        else:
            # Center x and center y coordinates
            nodule_diameter = csv_row[1]["Diameter [mm]"]
            cx, cy = int(x1 + w / 2), int(y1 + h / 2)
            x1, y1, x2, y2 =get_mm_box(centre_pixel=[cx, cy], bbox_size_in_mm=nodule_diameter,
                                                              pixel_spacing=np.array(image.GetSpacing()[0:2]))
        n_slices = get_n_slices(z_in_mm=nodule_diameter, pixel_spacing_z=image.GetSpacing()[2])
        bbox_medsam = [x1, y1, x2, y2]
        for i in range(len(image_array)):
            if (i >= (csv_slice_idx - int(n_slices / 2))) and (i <= (csv_slice_idx + int(n_slices / 2))):
                medsam_seg, box_np = medsam_segmentation(model_path=model_path, image=image_array[i],
                                                         bbox=bbox_medsam, device=device)
                processed_slices.append(medsam_seg)
            else:
                image_slice_no_prediction = np.zeros_like(image_array[i])
                processed_slices.append(image_slice_no_prediction)
        # Stack all the segmented 2D masks to form a 3D array
        medsam_segmented_volume = np.stack(processed_slices, axis=0)
        # Convert the 3D numpy array back to a SimpleITK image
        segmented_image = sitk.GetImageFromArray(medsam_segmented_volume)
        segmented_image.CopyInformation(image)
        # Save mask in specific output folder
        segmentation_filename = csv_row[1]["Segmentation filename"].replace(".mhd", "")
        sitk.WriteImage(segmented_image, str(Path(output_path) / str(segmentation_filename + ".mha")))
        logger.success(f"Inference for scan {scan_name} finished and saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Inference MedSAM')
    parser.add_argument("--input", type=str, help="Path to the folder containing scans for inference.")
    parser.add_argument("--output", type=str, help="Path to the output folder containing inference results.")
    parser.add_argument("--model", type=str, help="Path to the folder containing the model weights to be used for "
                                                  "inference.")
    parser.add_argument("--csv", type=str, help="Path to the csv file containing nodule locations.")
    parser.add_argument("--device", type=str, default='cpu', help="Device can be 'cpu', 'cuda', 'mps'")
    parser.add_argument("--bbox-mm", type=int, default=None, help="Bbox size in mm to be used for inference")

    option = parser.parse_args()
    input = option.input
    output = option.output
    model = option.model
    csv = option.csv
    bbox_mm = option.bbox_mm
    device = option.device

    run_inference(input_path=input, csv_path=csv, output_path=output, model_path=model, device=device, bbox_mm=bbox_mm)
