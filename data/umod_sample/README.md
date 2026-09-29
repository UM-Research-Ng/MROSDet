# UMOD demonstration subset

This directory contains **100 paired optical-sonar samples** selected from the
authors' UMOD_v2 dataset: 70 training pairs, 20 validation pairs, and 10 test
pairs. These are demonstration resources for the MROSDet implementation, not the
complete RUMOD benchmark or the complete test set used in the manuscript.

The original train/val/test membership is retained. Selection uses seed 0,
prioritizes rare-class coverage in each modality, and includes all nine classes
across the subset in both RGB and Sonar. The smaller test split is selected first,
then validation and training, without transferring source images between splits.
Exact duplicate image contents are excluded from the selected set. This is a
small class-aware sample, not a claim of representative benchmark sampling.

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

`manifest.json` records relative paths, SHA-256 checksums for each image and
label, dimensions, class coverage, and background-label status. `summary.json`
records valid-source/selected class counts and exact-content duplicate audit
counts. Complete source-duplicate path lists are not part of the public sample. It
contains no server absolute paths or training logs. The preparation helper
audits pairing, all source labels, image decoding, and checksums before
creating a new output directory. Missing images/labels, malformed labels,
undecodable images, and pairs containing identical optical and sonar image
content are excluded and recorded in the audit. No original annotation is
rewritten to make it pass validation. The selected subset must still satisfy
70/20/10 pairs and nine-class coverage in each modality. Normalized boxes must
have positive width/height and lie inside the image, allowing a 1e-5 rounding
tolerance at the boundary. The selected images and labels are copied
without recompression or annotation conversion.

Exact hashes cannot detect visually similar frames. Users extending this sample
should check capture/sequence identity when designing an independent evaluation.
The ten-pair test split is for execution checks, not stable model comparisons.

## License and attribution

Copyright (c) 2026 MROSDet authors.

These example images and annotations are licensed under **Creative Commons
Attribution 4.0 International (CC BY 4.0)**; see [LICENSE](LICENSE) and
<https://creativecommons.org/licenses/by/4.0/>. Give appropriate credit, link to
the license, and indicate any changes you make. The code repository uses a
different license; this data license applies to the example images and labels.

Suggested credit:

> UMOD demonstration subset, provided by the MROSDet authors (Mingxin Liu,
> Yujie Wu, Ruixin Li, Ziliang Ji, and Cong Lin), via
> https://github.com/UM-Research-Ng/MROSDet, licensed under CC BY 4.0.

Related manuscript: *MROSDet: A Modality-Robust Optical-Sonar Fusion Paradigm for
Underwater Perception under Sensor Degradation*. Paper link: to be added after
acceptance.
