"""CubeGB Studio — an all-in-one local web GUI.

Select an image → generate a ``.cgb`` → view it in 3D → export to glTF/OBJ, all
in one page. The backend (:mod:`app.server`) is a small FastAPI app that reuses
the recognition pipeline and the mesh baker; the frontend reuses CubeGB's
three.js renderer.

Run it with::

    python -m app.server          # or: cubegb-studio
"""
