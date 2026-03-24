import os
import mne
import numpy as np

from config import *
from scipy.io import loadmat


def load_eeg_data(file_path):
    """
    Load EEG data from various file formats.

    Supported formats: .mat, .cnt, .npy, .npz, .edf, .bdf

    Parameters:
    file_path (str): Path to the EEG data file.

    Returns:
    data: Loaded EEG data. Type depends on the file format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".mat":
        # Load MATLAB .mat file
        data = loadmat(file_path)
        return data
    elif file_extension == ".cnt":
        # Load Neuroscan .cnt file using MNE
        raw = mne.io.read_raw_cnt(file_path, preload=True)
        return raw
    elif file_extension in [".npy", ".npz"]:
        # Load NumPy .npy or .npz file
        data = np.load(file_path)
        return data
    elif file_extension == ".edf":
        # Load European Data Format .edf file using MNE
        raw = mne.io.read_raw_edf(file_path, preload=True)
        return raw
    elif file_extension == ".bdf":
        # Load BioSemi Data Format .bdf file using MNE
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        return raw
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Supported formats: .mat, .cnt, .npy, .npz, .edf, .bdf"
        )


def load_all_eeg_data_in_folder(folder_path):
    """
    Load all EEG data files in a folder.

    Parameters:
    folder_path (str): Path to the folder containing EEG data files.

    Returns:
    dict: Dictionary with file names as keys and loaded data as values.
    """
    data_dict = {}

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if (
            os.path.isfile(file_path)
            and os.path.splitext(file_name)[1].lower() in ALLOWED_EXTENSIONS
        ):
            try:
                data = load_eeg_data(file_path)
                data_dict[file_name] = data
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    return data_dict
