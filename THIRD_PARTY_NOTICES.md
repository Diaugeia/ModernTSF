# Third-Party Notices

ModernTSF maintains every registered model as local repository code. Papers and
official repositories are cited to document provenance and implementation
decisions; those references do not make external model source part of this
distribution.

Each model card at `src/moderntsf/models/<model>/README.md` records the paper and, when an
official implementation exists, its repository URL, pinned revision, and license
label. Consult that repository at the recorded revision for its complete license
and notices. A model card with `codebase: null` has no identified official
codebase.

Runtime Python dependencies are declared in `pyproject.toml` and retain their own
licenses and notices. ModernTSF does not vendor their source. Built wheels contain
the ModernTSF runtime packages and curated Agent assets, not external model
repositories, test suites, local datasets, or downloaded model artifacts.
