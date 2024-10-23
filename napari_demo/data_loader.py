from pathlib import Path
import nibabel as nib
import numpy as np

DATA_PATH = "c3ro_dataset"

def get_available_ana_region(data_path:str=DATA_PATH) -> list:
    
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError("C3ro data path does not exist")
    
    available_ana_region = []
    for p in data_path.glob("*"):
        if p.is_dir():
            available_ana_region.append(p.stem)

    return available_ana_region


def get_available_roi(data_path:str=DATA_PATH, ana_region:str="H&N") -> list:
    
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError("C3ro data path does not exist")

    data_path = data_path.joinpath(ana_region)

    if not data_path.exists():
        raise FileNotFoundError("Anatomical region does not exist")
    
    segs_path = data_path.joinpath("Segmentations/Expert/Consensus")    

    available_roi = set()

    for p in segs_path.rglob("*.gz"):
        aroi = p.stem[:-4]
        aroi = aroi.replace("_STAPLE", "")
        available_roi.add(aroi)
    
    return list(available_roi)


def load_case(data_path:str=DATA_PATH, ana_region:str="H&N", roi:str="Parotid_L", use_expert_segs:bool=False) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError("C3ro data path does not exist")

    data_path = data_path.joinpath(ana_region)

    if not data_path.exists():
        raise FileNotFoundError("Anatomical region does not exist")

    img_path = data_path.joinpath(f"CT/NIFTI/Image_CT_{ana_region}.nii.gz")
    
    if use_expert_segs:
        segs_path = data_path.joinpath("Segmentations/Expert")
    else:
        segs_path = data_path.joinpath("Segmentations/Non-Expert")    

    if len(list(segs_path.rglob(f"*_{roi}*"))) <= 0 or len(list(segs_path.rglob(f"*_{roi}*"))) <= 0:
        raise FileNotFoundError("ROI does not exist")

    #########
    # IMAGE #
    #########

    # Load the NIfTI image
    nifti_img = nib.load(img_path)

    # Get the image data as a NumPy array
    img = nifti_img.get_fdata()

    #################
    # SEGMENTATIONS #
    #################

    roi_segs_paths = list(segs_path.rglob(f"*_{roi}*"))
    roi_consensus_path = list(segs_path.rglob(f"{roi}_STAPLE*"))[0]

    segs = []
    for seg_path in roi_segs_paths:
        nifti_seg = nib.load(seg_path)
        seg_data = nifti_seg.get_fdata()
        seg_data = seg_data.astype(int)
        segs.append(seg_data)

    consensus = nib.load(roi_consensus_path).get_fdata().astype(int)

    ############
    # FIX DIMS #
    ############

    img = img.transpose(2,1,0)
    consensus = consensus.transpose(2,1,0)
    for i, seg in enumerate(segs):
        segs[i] = seg.transpose(2,1,0)

    return img, consensus, segs