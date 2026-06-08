'''
    Copyright (c) 2026 Chenxi Hu, Yichen Zhao, and Haiyang Chen, SJTU 2026.
    All rights reserved.

    This software is released under a custom Research-Only License.
    It is provided solely for academic research purpose related to the manuscript.

    Commercial use, clinical use, redistribution, sublicensing, or use in
    commercial product development is not permitted without prior written
    permission from the copyright holders.

    Citation:
    If you use this code or any part of this repository, please cite:
    [Citation information to be updated after publication]
'''

import os
import re
import copy
import argparse

import pydicom
import numpy as np
import torch
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import normalization
from CombinedNetwork import CombinedNet
from strain import strain_calculation


def extract_number(filename: str):
    """Sort DICOM files by the last number in filename."""
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else 0


def read_heart_rate_from_dicom(ds):
    """
    Read heart rate from a DICOM dataset.

    Priority:
    1. HeartRate / (0018,1088): beats per minute.
    2. NominalInterval / (0018,1062): R-R interval in ms, HR = 60000 / interval.

    Returns None if unavailable.
    """
    # Standard DICOM keyword: HeartRate, tag (0018,1088)
    hr = getattr(ds, "HeartRate", None)
    if hr is None and (0x0018, 0x1088) in ds:
        hr = ds[(0x0018, 0x1088)].value

    if hr is not None:
        try:
            # Some DICOM fields may be MultiValue or string-like.
            if isinstance(hr, (list, tuple)):
                hr = hr[0]
            return float(hr)
        except Exception:
            pass

    # Standard DICOM keyword: NominalInterval, tag (0018,1062), unit: ms
    nominal_interval = getattr(ds, "NominalInterval", None)
    if nominal_interval is None and (0x0018, 0x1062) in ds:
        nominal_interval = ds[(0x0018, 0x1062)].value

    if nominal_interval is not None:
        try:
            if isinstance(nominal_interval, (list, tuple)):
                nominal_interval = nominal_interval[0]
            nominal_interval = float(nominal_interval)
            if nominal_interval > 0:
                return 60000.0 / nominal_interval
        except Exception:
            pass

    return None

def read_pixel_spacing_from_dicom(ds):
    """
    Read in-plane pixel spacing from DICOM.

    Priority:
    1. PixelSpacing / (0028,0030): [row_spacing, col_spacing], unit: mm
    2. ImagerPixelSpacing / (0018,1164), if PixelSpacing is unavailable
    3. Fallback to [1.0, 1.0]
    """
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is None and (0x0028, 0x0030) in ds:
        spacing = ds[(0x0028, 0x0030)].value

    if spacing is None:
        spacing = getattr(ds, "ImagerPixelSpacing", None)
        if spacing is None and (0x0018, 0x1164) in ds:
            spacing = ds[(0x0018, 0x1164)].value

    if spacing is not None:
        try:
            return [float(spacing[0]), float(spacing[1])]
        except Exception:
            pass

    print("[Warning] Pixel spacing was not found in DICOM. Falling back to [1.0, 1.0].")
    return [1.0, 1.0]

def process_dicom(folder_path: str, target_x: float = 1.5, target_z: int = 25, crop_size: int = 160):
    """
    Convert a folder of single-frame DICOMs into normalized tensor [1, H, W, T].
    Also tries to read HR from the first valid DICOM.
    """
    dcm_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
    dcm_files.sort(key=extract_number)

    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No .dcm files found in: {folder_path}")

    sequence = []
    hr = None

    for i, file in enumerate(dcm_files):
        dcm_path = os.path.join(folder_path, file)
        ds = pydicom.dcmread(dcm_path)

        if hr is None:
            hr = read_heart_rate_from_dicom(ds)

        img = ds.pixel_array.astype(np.float32)

        spacing = read_pixel_spacing_from_dicom(ds)
        scale_factors = [float(spacing[0]) / target_x, float(spacing[1]) / target_x]
        resized = zoom(img, scale_factors, order=1)

        h, w = resized.shape
        start_h, start_w = (h - crop_size) // 2, (w - crop_size) // 2

        if start_h < 0 or start_w < 0:
            raise ValueError(
                f"Resized image is smaller than crop_size={crop_size}: got {resized.shape}. "
                "Please reduce crop_size or check PixelSpacing/target_x."
            )

        cropped = resized[start_h:start_h + crop_size, start_w:start_w + crop_size]
        sequence.append(cropped)

    sequence = np.array(sequence)  # [T, H, W]
    current_z = sequence.shape[0]
    depth_factor = target_z / current_z
    sequence = zoom(sequence, (depth_factor, 1, 1), order=3)  # [target_z, H, W]

    sequence = torch.from_numpy(sequence).float().permute(1, 2, 0).unsqueeze(0)  # [1, H, W, T]
    sequence = normalization(sequence)

    if hr is None:
        print("[Warning] Heart rate was not found in DICOM. Falling back to HR=60.")
        hr = 60.0

    return sequence, hr


def supertrans(x, y, U):
    """Bilinear interpolation for displacement field U."""
    Nx, Ny = U.shape
    lx = int(np.floor(x))
    ly = int(np.floor(y))
    rx = lx + 1 if lx + 1 < Nx else lx
    ry = ly + 1 if ly + 1 < Ny else ly

    trans = (
        U[rx, ry] * (x - lx) * (y - ly)
        + U[rx, ly] * (x - lx) * (ry - y)
        + U[lx, ry] * (rx - x) * (y - ly)
        + U[lx, ly] * (rx - x) * (ry - y)
    )
    return trans


def load_model(model_path: str, size: int, device: torch.device):
    netRegis = CombinedNet([size, size], True, True).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Remove grid buffers if present, same as the original script.
    keys_to_remove = [key for key in checkpoint.keys() if "grid" in key]
    for key in keys_to_remove:
        del checkpoint[key]

    netRegis.load_state_dict(checkpoint, strict=False)
    netRegis.eval()
    return netRegis


def run_pipeline(
    dicom_folder: str,
    model_path: str,
    output_dir: str,
    idx: str = "4ch",
    target_x: float = 1.5,
    target_z: int = 25,
    crop_size: int = 160,
    save_processed: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    sequence, hr = process_dicom(
        dicom_folder,
        target_x=target_x,
        target_z=target_z,
        crop_size=crop_size,
    )

    print(f"Processed tensor shape: {tuple(sequence.shape)}")
    print(f"Heart rate used for strain calculation: {hr:.2f} bpm")

    if save_processed:
        processed_path = os.path.join(output_dir, f"{idx}_processed.pt")
        torch.save(sequence, processed_path)
        print(f"Saved processed tensor to: {processed_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = sequence.shape[1]

    netRegis = load_model(model_path, size, device)

    # Original model expects [B, C, H, W, T], while process_dicom returns [C, H, W, T].
    testdata = torch.unsqueeze(sequence, 0).to(device)

    with torch.no_grad():
        PredictedMask, WrapMask, PosImage, NegV, V = netRegis(testdata)

    PredictedMask = PredictedMask.cpu().detach().numpy()
    PredictedMask = np.argmax(PredictedMask, 1)
    NegV = NegV.cpu().detach().numpy()
    V = V.cpu().detach().numpy()

    fir = PredictedMask[0, 0]

    # Calculate strain and strain rate.
    strain_calculation(fir, V, NegV, idx, hr, size)

    # Generate segmentation mask of the first frame.
    Image = copy.deepcopy(testdata)
    Image = torch.squeeze(Image, 0)
    Image = torch.squeeze(Image, 0)
    Image = Image.cpu().detach().numpy()
    Image = np.transpose(Image, (2, 0, 1))  # [T, H, W]
    Image0 = Image[0, :, :]

    seg_path = os.path.join(output_dir, f"segmentation_{idx}.jpg")
    plt.figure(figsize=(5, 5))
    plt.imshow(Image0, cmap="gray")
    plt.imshow(fir, cmap="Reds", alpha=0.2)
    plt.axis("off")
    plt.savefig(seg_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()
    print(f"Saved segmentation preview to: {seg_path}")

    # Generate tracking gif.
    coords = np.argwhere(fir == 1)
    Back = []
    for i in range(len(coords)):
        x = coords[i][0]
        y = coords[i][1]
        x_regis = x + supertrans(x, y, NegV[0, 1, 0])
        y_regis = y + supertrans(x, y, NegV[0, 0, 0])
        back = []
        for j in range(target_z):
            x_back = x_regis + supertrans(x_regis, y_regis, V[0, 1, j])
            y_back = y_regis + supertrans(x_regis, y_regis, V[0, 0, j])
            back.append([y_back, x_back])
        Back.append(back)

    Back = np.transpose(Back, (1, 0, 2)) if len(Back) > 0 else np.zeros((target_z, 0, 2))

    denom = np.max(Image[:]) - np.min(Image[:])
    if denom == 0:
        image_norm = np.zeros_like(Image)
    else:
        image_norm = (Image - np.min(Image[:])) / denom

    image_rgb = np.zeros((target_z, size, size, 3), dtype=np.uint8)
    for i in range(3):
        image_rgb[:, :, :, i] = image_norm * 255

    fig, ax = plt.subplots(dpi=100)
    im = ax.imshow(image_rgb[0], cmap="gray")
    scatter = ax.scatter([], [], color="c", s=1, marker="x")
    ax.axis("off")

    def init():
        return im, scatter

    def update(frame):
        im.set_data(image_rgb[frame])
        scatter.set_offsets(Back[frame])
        return im, scatter

    gif_path = os.path.join(output_dir, f"tracking_{idx}.gif")
    anim = FuncAnimation(fig, update, frames=range(target_z), init_func=init, blit=True, interval=100)
    anim.save(gif_path, writer="pillow", dpi=100)
    plt.close("all")
    print(f"Saved tracking gif to: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM preprocessing + LA strain calculation pipeline.")
    parser.add_argument("--dicom_folder", type=str, default="./examples/4ch_anonymized", help="Folder containing DICOM files.")
    parser.add_argument("--model_path", type=str, default="./netJoint.pkl", help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory.")
    parser.add_argument("--idx", type=str, default="4ch", help="View name, e.g., 2ch or 4ch.")
    parser.add_argument("--target_x", type=float, default=1.5, help="Target in-plane spacing.")
    parser.add_argument("--target_z", type=int, default=25, help="Target number of cine frames.")
    parser.add_argument("--crop_size", type=int, default=160, help="Center crop size.")
    parser.add_argument("--no_save_processed", action="store_true", help="Do not save processed .pt tensor.")

    args = parser.parse_args()

    run_pipeline(
        dicom_folder=args.dicom_folder,
        model_path=args.model_path,
        output_dir=args.output_dir,
        idx=args.idx,
        target_x=args.target_x,
        target_z=args.target_z,
        crop_size=args.crop_size,
        save_processed=not args.no_save_processed,
    )
