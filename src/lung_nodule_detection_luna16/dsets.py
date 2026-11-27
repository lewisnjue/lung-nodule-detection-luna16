from collections import namedtuple
import copy
import csv
import functools
from pathlib import Path

import diskcache
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from lung_nodule_detection_luna16.utils import IrcTuple, XyzTuple, irc2xyz, xyz2irc

raw_cache = diskcache.FanoutCache("data-unversioned/cache/luna_raw")

CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple",
    "isNodule_bool, diameter_mm, series_uid, center_xyz",
)
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_FOLDER = Path(f"{ROOT_DIR}/data")
DEFAULT_MHD_DATA_FOLDER = Path(f"{ROOT_DIR}/data")


@functools.lru_cache(maxsize=1)
def getCandidateInfoList(
    require_on_disk=True,
    mhd_data_folder=DEFAULT_MHD_DATA_FOLDER,
    annotations_csv=Path(f"{ROOT_DIR}/data/annotations.csv"),
    candidates_csv=Path(f"{ROOT_DIR}/data/candidates.csv"),
):
    data_directory = Path(mhd_data_folder)
    mhd_files = list(data_directory.rglob("subset*/*.mhd"))
    if not mhd_files:
        print(f"Warning: No .mhd files found under {data_directory}")

    present_on_disk_set = {path.stem for path in mhd_files}
    diameter_dict = {}

    with open(annotations_csv, "r", newline="") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple(float(x) for x in row[1:4])
            annotationDiameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []
    with open(candidates_csv, "r", newline="") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if require_on_disk and series_uid not in present_on_disk_set:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple(float(x) for x in row[1:4])
            candidateDiameter_mm = 0.0

            for annotationCenter_xyz, annotationDiameter_mm in diameter_dict.get(
                series_uid, []
            ):
                within_bounds = all(
                    abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    <= annotationDiameter_mm / 4
                    for i in range(3)
                )
                if within_bounds:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(
                CandidateInfoTuple(
                    isNodule_bool,
                    candidateDiameter_mm,
                    series_uid,
                    candidateCenter_xyz,
                )
            )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid, mhd_data_folder=DEFAULT_MHD_DATA_FOLDER):
        data_directory = Path(mhd_data_folder)
        matches = list(data_directory.rglob(f"subset*/{series_uid}.mhd"))
        if not matches:
            raise FileNotFoundError(
                f"No .mhd file found for series_uid={series_uid!r} under {data_directory}"
            )

        ct_mhd = sitk.ReadImage(str(matches[0]))
        self.series_uid = series_uid
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, out=ct_a)
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        pad_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            # Clamp to valid array bounds and record required padding
            pad_before = max(0, -start_ndx)
            pad_after = max(0, end_ndx - self.hu_a.shape[axis])

            clamped_start = max(0, start_ndx)
            clamped_end = min(self.hu_a.shape[axis], end_ndx)

            slice_list.append(slice(clamped_start, clamped_end))
            pad_list.append((pad_before, pad_after))

        ct_chunk = self.hu_a[tuple(slice_list)]
        if any(before > 0 or after > 0 for before, after in pad_list):
            ct_chunk = np.pad(
                ct_chunk,
                pad_width=pad_list,
                mode="constant",
                constant_values=-1000.0,
            )

        return ct_chunk, center_irc


@functools.lru_cache(maxsize=1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        isValSet_bool=None,
        series_uid=None,
    ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a).to(torch.float32).unsqueeze(0)
        pos_t = torch.tensor(
            [
                not candidateInfo_tup.isNodule_bool,
                candidateInfo_tup.isNodule_bool,
            ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )
