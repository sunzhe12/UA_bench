# Data

UA-Bench's 375,051 audio records are **not** stored in git. The corpus is far
too large for a repository — GitHub caps individual files at 100 MB and a
repository should stay well under 1 GB.

Host the release externally and link it from here:

- **Hugging Face Datasets** — lets people run
  `load_dataset("sunzhe12/UA-Bench")` and preview clips in the browser. This is
  the default choice in the audio/ML community.
- **Zenodo** — gives you a DOI, which is what you want if the paper needs a
  citable, permanent link.

Keep only small artifacts in this directory: the 2,600-record evaluation subset
metadata, label hierarchy definitions, or a `download.sh` that pulls the corpus
from its external host.
