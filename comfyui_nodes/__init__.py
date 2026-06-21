"""CubeGB ComfyUI custom-node package (Phase 6).

ComfyUI discovers custom nodes by importing this package and reading two module
-level dicts:

* ``NODE_CLASS_MAPPINGS``        -- internal id -> node class
* ``NODE_DISPLAY_NAME_MAPPINGS`` -- internal id -> human label (shown in the UI)

To install, clone the CubeGB repo into ``ComfyUI/custom_nodes/`` so the path is
``ComfyUI/custom_nodes/cubegb/comfyui_nodes/`` and restart ComfyUI. The node
pack loads even when the heavy recognition dependencies (torch / SAM / depth
models) are absent -- those are imported lazily and only the *Generate* node
requires them at run time.

The four nodes appear under the **CubeGB** category:

* **CubeGB Generate** -- IMAGE -> CGB (runs the recognition pipeline)
* **CubeGB Save**     -- CGB -> path (pretty-JSON ``.cgb`` in the output dir)
* **CubeGB Bake**     -- CGB -> path (glb/gltf/obj mesh in the output dir)
* **CubeGB Preview**  -- CGB -> stats STRING + preview IMAGE
"""

from __future__ import annotations

from .nodes import CubeGBBake, CubeGBGenerate, CubeGBPreview, CubeGBSave

# Internal ids are namespaced to avoid collisions with other custom-node packs.
NODE_CLASS_MAPPINGS: dict[str, type] = {
    "CubeGBGenerate": CubeGBGenerate,
    "CubeGBSave": CubeGBSave,
    "CubeGBBake": CubeGBBake,
    "CubeGBPreview": CubeGBPreview,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "CubeGBGenerate": "CubeGB Generate",
    "CubeGBSave": "CubeGB Save",
    "CubeGBBake": "CubeGB Bake",
    "CubeGBPreview": "CubeGB Preview",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
