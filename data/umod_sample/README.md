# UMOD demonstration subset

This directory contains **100 optical-sonar image pairs** from UMOD_v2:
70 training pairs, 20 validation pairs, and 10 test pairs. Each split covers all
nine classes in both modalities. The subset is intended for running the MROSDet
examples; it does not replace the complete benchmark used in the paper.

## Layout and labels

```text
RGB/{train,val,test}/{images,labels}/
Sonar/{train,val,test}/{images,labels}/
manifest.json
summary.json
```

Match optical and sonar images by their relative path without the extension;
the two modalities may use different image formats. Each modality has its own
`labels/<stem>.txt` file. Each nonempty line contains five fields:

```text
class_id center_x center_y width height
```

Class IDs are zero-based, with names defined in
[`configs/umod_sample.yaml`](../../configs/umod_sample.yaml) at the repository
root: cage, frame, hook, anchor, tire, rov, plastic bucket, fish, and oil drums
(IDs 0 through 8 respectively). Box coordinates are normalized against that modality's
original image dimensions. An existing empty label means that no object is
annotated. A missing label is not treated as an empty label.

Boxes must have positive width and height and lie inside the image, with a
`1e-5` rounding tolerance at the boundary. `manifest.json` records each image
and label's relative path, SHA-256 checksum, dimensions, and class information.
`summary.json` contains class counts and duplicate-audit counts.

## Subset selection

Selection uses seed 0 and preserves the original train/val/test membership.
The smaller test split is selected first, followed by validation and training,
prioritizing class coverage in both modalities. Image files with identical
SHA-256 hashes are excluded from the selected pairs. Images and labels are copied without
recompression or annotation conversion.

With access to the complete UMOD_v2 source, run this command from the repository
root to prepare the subset in a new directory:

```bash
python tools/prepare_sample.py --source <UMOD-root> --destination <new-directory>
```

The destination must not exist. The script checks pairs, labels, image decoding,
and content hashes before copying. It excludes pairs with missing files, invalid
labels, undecodable images, or identical optical/sonar image files, and reports
the exclusions. Selection fails if the required counts or class coverage cannot
be met. When creating additional splits, group related video frames or captures
by sequence: file hashes alone cannot identify visually similar observations.

## License and attribution

Copyright (c) 2026 MROSDet authors.

These images and annotations are licensed under **Creative Commons Attribution
4.0 International (CC BY 4.0)**; see [LICENSE](LICENSE) and
<https://creativecommons.org/licenses/by/4.0/>. Please credit the authors, link to
the license, and indicate any changes.

Suggested credit:

> UMOD demonstration subset, provided by the MROSDet authors (Mingxin Liu,
> Yujie Wu, Ruixin Li, Ziliang Ji, and Cong Lin), via
> https://github.com/UM-Research-Ng/MROSDet, licensed under CC BY 4.0.

Related manuscript: *MROSDet: A Modality-Robust Optical-Sonar Fusion Paradigm for
Underwater Perception under Sensor Degradation*. Paper link: to be added after
acceptance.
