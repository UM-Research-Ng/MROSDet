# Third-party notices

## Ultralytics

Network layers and graph construction, bounding-box operations, detection
assignment and losses, detection metrics, and training utilities contain code
adapted from Ultralytics. The MROSDet research implementation was originally
developed within that framework.

- Upstream project: <https://github.com/ultralytics/ultralytics>
- Upstream copyright: Copyright (c) Ultralytics, as specified in the retained
  source headers.
- Upstream license: GNU Affero General Public License, version 3.0 (AGPL-3.0).
- License text: [LICENSE](LICENSE).

Adaptation date: **2026-09-29**. Required components were extracted into
`mrosdet`; unrelated tasks and services were removed. The adaptation adds paired
data entry points, explicit weight loading, and a parameter-based checkpoint
format. It preserves MROSDet's reliability estimation, cross-modal fusion,
dual-pyramid processing, and bimodal detection computation.

Documentation and explanatory comments were revised on **2026-09-30**. Original
copyright and license notices remain in the adapted source files. Those notices
and the applicable AGPL-3.0 terms continue to apply to redistributed versions.

## Runtime dependencies

Dependencies declared in `pyproject.toml` are distributed separately under their
respective licenses; consult their package metadata and upstream projects for
the corresponding license texts.

## Example data and model checkpoint

The example UMOD image pairs and annotations are licensed separately under
[Creative Commons Attribution 4.0 International](data/umod_sample/LICENSE).
Their credit and provenance are provided in the
[dataset README](data/umod_sample/README.md).

The released MROSDet checkpoint is covered by AGPL-3.0. It was converted from
the authors' designated trained model; its metadata retains the source-file
SHA-256 value. Other pretrained checkpoints are not included.
