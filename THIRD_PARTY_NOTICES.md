# Third-party notices

## Ultralytics

Parts of the model infrastructure in this repository are derived from the
Ultralytics implementation: network layers and graph construction, bounding-box
operations, detection assignment and losses, detection metrics, and training
utilities. The MROSDet research implementation was originally developed within
that framework.

- Upstream project: <https://github.com/ultralytics/ultralytics>
- Upstream copyright: Copyright (c) Ultralytics, as specified in the retained
  source headers.
- Upstream license: GNU Affero General Public License, version 3.0 (AGPL-3.0).
- License text: [LICENSE](LICENSE).

For this release, the necessary components have been extracted into `mrosdet`
and adapted for a standalone optical-sonar detector. Changes include removing
unrelated tasks and services, restricting execution to the supported paired-image
workflow, replacing framework-specific public entry points, making weight loading
explicit, and replacing executable-object checkpoints with an independently
loadable parameter format. MROSDet's reliability estimation, reliability-guided
fusion, dual-pyramid processing, and two-branch detection logic are retained.

The package does not import or require the `ultralytics` package at runtime.
That packaging change does not imply that the adapted code has no upstream
origin. Applicable copyright and license notices must be retained when
redistributing this code or modifications. This release is not affiliated with
or endorsed by Ultralytics.

## Runtime dependencies

PyTorch, torchvision, NumPy, Pillow, PyYAML, and other dependencies declared in
`pyproject.toml` are distributed separately under their respective licenses.
Installation does not transfer ownership or replace those license terms.
Consult the installed distribution metadata and upstream projects for their
license texts. Dependency source trees are not bundled in this repository.

## Example data and model checkpoint

The example UMOD image pairs and annotations are licensed separately under
[Creative Commons Attribution 4.0 International](data/umod_sample/LICENSE).
Their credit and provenance are provided in the
[dataset README](data/umod_sample/README.md).

The distributed MROSDet model checkpoint is covered by the repository's
AGPL-3.0 license. It is converted from the authors' designated trained MROSDet
checkpoint, and its metadata records the source-file SHA-256 value. Conversion
does not make it a checkpoint of any other model. No third-party pretrained
checkpoint is bundled in the repository or published as a release asset.
