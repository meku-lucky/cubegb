"""Per-object reconstruction tests (recognition.object_recon).

Uses synthetic 2D masks (no SAM / model weights). Skips without OpenCV/scipy.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

import cgb
from recognition.object_recon import reconstruct_object, object_to_documents


def _disc(n=120, r=40):
    yy, xx = np.mgrid[0:n, 0:n]
    return ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) <= r ** 2


def test_silhouette_extrude_is_thin_in_depth():
    occ, colmap = reconstruct_object(_disc(), res=48, depth_frac=0.3, dome=True)
    assert occ.sum() > 0
    idx = np.argwhere(occ)
    ext = idx.max(0) - idx.min(0)
    # extruded disc: depth (z) thinner than the in-plane x/y span
    assert ext[2] < ext[0] and ext[2] < ext[1]


def test_empty_mask_is_safe():
    occ, colmap = reconstruct_object(np.zeros((40, 40), bool), res=32)
    assert occ.sum() == 0 and colmap == {}


def test_object_documents_validate():
    occ, colmap = reconstruct_object(_disc(), res=48, rgb=None)
    prim, vox = object_to_documents(occ, colmap, max_prims=4)
    cgb.validate(prim)
    cgb.validate(vox)
    assert len(prim["primitives"]) >= 1
    assert len(vox["primitives"]) == int(occ.sum())


def test_dome_is_thicker_at_centre_than_flat():
    occ_dome, _ = reconstruct_object(_disc(), res=48, depth_frac=0.4, dome=True)
    occ_flat, _ = reconstruct_object(_disc(), res=48, depth_frac=0.4, dome=False)
    # a flat slab fills more voxels than a tapering dome of the same depth budget
    assert occ_flat.sum() > occ_dome.sum()
