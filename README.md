# contour-depth

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/contour-depth.svg)](https://badge.fury.io/py/contour-depth)

Python library for computing statistical depth of ensembles of contours. The library supports the Contour Band Depth and Inclusion Depth methods. It also supports finding the ensemble's modes of variation by using depth-based clustering. Finally, it offers visualization utilities like spaghetti plots and Contour Box Plots. 

## Installation

You can install the library via pip:

```bash
pip install contour-depth
```

## Usage

To setup an environment follow the steps:
1. Install a conda (we recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/))
2. Create environment: `conda create --name=test-env python=3.9`
3. Activate environment: `conda activate test-env`
4. Install dependencies with pip: `pip install contour-depth` and `pip install matplotlib`. Other dependencies should be already available.
5. To test installation, from the root of the repository run `python visualize.py`. No errors should be raised.

```python
from contour_depth import Depth, Metric
import contour_depth
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import time

if __name__ == "__main__":
  masks = contour_depth.synthetic_data.three_rings(num_masks=250, num_rows=256, num_cols=256, seed=42)
  start = time.time()
  #clusters = contour_depth.cluster_inclusion_matrix(masks=masks, num_clusters=3, depth=Depth.EpsilonInclusionDepth, metric=Metric.Depth, kmeans_random_seed=42)
  clusters = contour_depth.cluster_optimized_eid(masks=masks, num_clusters=3, metric=Metric.Depth, kmeans_random_seed=42)
  print(f"{(time.time() - start)*1000:.2f} ms to compute clusters")

  height, width = masks[0].shape
  out_image = np.zeros((height, width, 3), dtype=np.float32)
  for mask, cluster in zip(masks, clusters):
    grown_mask = scipy.signal.convolve2d(mask, np.ones((3, 3)), mode="same") > 0
    mask_edge = grown_mask - mask
    color = np.random.default_rng(cluster).random(3)
    out_image += np.multiply.outer(mask_edge, color)

  fig, ax = plt.subplots(1, 1)
  ax.imshow(out_image / np.max(out_image))
  plt.show()
```

## Citation

If you use this library in your work and would like to cite it, please use the following BibTeX entry:

```bibtex
@misc{author_year_libraryname,
  author = {Author Name},
  title = {Library Name},
  year = {Year},
  publisher = {Publisher},
  howpublished = {\url{link-to-paper}},
}
```

## License

This project is licensed under the terms of the [MIT license](https://github.com/chadepl/contour-depth/blob/main/LICENSE).
