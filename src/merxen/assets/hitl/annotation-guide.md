# Cellpose annotation pilot — guide pilot-v1

These crops are for developing a consistent correction style. They are marked
`pilot_train`; they must never be used as a held-out test set. Automatic crops
use image signal and spatial coverage. Viewer-selected crops can include
specific segmentation failures; the manifest records the selection method.

## Open the package

Extract `annotation-pilot.tar.gz` into a new folder. Keep an untouched copy.
Open a root-level `.tif` in the Cellpose 4.2.1.1 GUI. Its matching `_seg.npy`
contains the current Cellpose masks. Use the images in filename order.
`context/overview.png` shows the crop locations in the native section.
The TIFF channels and their order are listed in `manifest.json` under
`view_channels`; the model channels can differ from this display choice.

The raw and normalized arrays in subfolders are for future server-side
training. Leave them unchanged. Image normalization in the GUI affects the
display; it does not replace the exported training input. Do not train in the GUI.

Crop locations can come from automatic tissue sampling or from regions you
selected in the MerXen viewer. The manifest records which method was used.
In either case, use Cellpose here to edit the cell outlines. The viewer's HITL
crop rectangles are location selections, not cell masks.

## Correct every visible cell

- Draw every confidently visible soma or nucleus, including partial cells at
  the crop edge. Remove false masks on neuropil, debris and background.
- A DAPI nucleus counts even when its cytoplasmic stain is weak. Include the
  contiguous perinuclear rim when visible; do not invent missing cytoplasm.
- Draw a compact, clearly bounded anucleate soma only when its stain is
  convincing and its diameter is at least about 5 µm.
- For neurons, include the soma and an unbroken proximal dendrite trunk up to
  half the soma diameter, capped at 10 µm. Exclude thin branches, distal
  processes and neuropil. Apply this rule consistently across crops.
- Separate touching nuclei using the DAPI boundary. Draw oligodendrocyte rows
  as individual cells. Draw vascular cells, not the vessel lumen.
- If a boundary remains ambiguous after about ten seconds, draw only the clear
  part and record the uncertainty in `review.tsv`.
- Use one connected component per cell and avoid overlaps. Delete and redraw
  a wrong mask rather than leaving fragments assigned to one cell.

## Save and return

Press **Ctrl+S** for every crop, even when no correction was needed. Only then
set that crop's `review.tsv` status to `reviewed`, and record minutes and notes.
Use `excluded` plus a reason for folds, tears, saturated images, wrong tissue or
unusable contrast. Leave unfinished crops `pending`. Keep crop IDs unchanged.

Return the corrected `_seg.npy` files and `review.tsv`, together with the
original manifest. GUI saves replace the original prefill checksums; preserve
the untouched package as the audit copy. Future import will validate image
identity, saved edits, connected labels and review status before training.

This release exports the pilot. Annotation import, model training and model
adoption are subsequent milestones; saving these crops does not launch them.
