# contour-depth

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/library-name.svg)](https://badge.fury.io/py/library-name)

Python library for computing statistical depth of ensembles of contours. The library supports the Contour Band Depth and Inclusion Depth methods. It also supports finding the ensemble's modes of variation by using depth-based clustering. Finally, it offers visualization utilities like spaghetti plots and Contour Box Plots. 

## Installation

You can install the library via pip:

```bash
pip install contour-depth
```

## Usage

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

## Documentation

For detailed usage instructions and API documentation, please refer to the [documentation](https://link-to-documentation).

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

This project is licensed under the terms of the [MIT license](LICENSE).
