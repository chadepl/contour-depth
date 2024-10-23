# contour-depth

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/contour-depth.svg)](https://badge.fury.io/py/contour-depth)

Python library for computing statistical depth of ensembles of contours. The library supports the Contour Band Depth and Inclusion Depth methods. It also supports finding the ensemble's modes of variation by using depth-based clustering. Finally, it offers visualization utilities like spaghetti plots and Contour Box Plots. 

![Different stages of the ensemble analysis process for an ensemble of segmentations of the right parotid gland of a head and neck cancer patient. a) and b) present an overview of the ensemble using a spaghetti plot and a contour boxplot based on the depths of the complete ensemble. c) and d) present a multi-modal analysis of the ensemble. c) depicts an overview of the different modes of variation and d) focuses on the less representative variation mode.](teaser-multimodal-contour-depth.png "Analysis pipeline using contour depths")

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
4. Install dependencies with pip: `pip install contour-depth` (or `pip install .` if building from the repository) and `pip install matplotlib`. Other dependencies should be already available.
5. To test installation, from the root of the repository run `python boxplot_demo.py` or `python clustering_demo.py`. No errors should be raised.

The directory `napari_demo` shows how to integrate the `contour-depth` package with a graphical user interface.
Further, it demonstrates the usage of the `contour-depth` package with three-dimensional data using a medical image segmentation dataset.

## Citation

If you use this library in your work and would like to cite it, please use the following BibTeX entries:

```bibtex
@article{10.1109/TVCG.2024.3350076,
  title={Inclusion Depth for Contour Ensembles}, 
  author={Chaves-de-Plaza, Nicolas F. and Mody, Prerak and Staring, Marius and van Egmond, René and Vilanova, Anna and Hildebrandt, Klaus},
  journal={IEEE Transactions on Visualization and Computer Graphics},   
  year={2024},
  volume={30},
  number={9},  
  pages={6560-6571},
  keywords={Data visualization;Visualization;Uncertainty;Feature extraction;Data models;Computational modeling;Semantic segmentation;Uncertainty visualization;contours;ensemble summarization;depth statistics},
  doi={10.1109/TVCG.2024.3350076}
}
```

```bibtex
@article{10.1111/cgf.15083,
  title = {Depth for Multi-Modal Contour Ensembles},
	author = {Chaves-de-Plaza, N.F. and Molenaar, M. and Mody, P. and Staring, M. and van Egmond, R. and Eisemann, E. and Vilanova, A. and Hildebrandt, K.},
	journal = {Computer Graphics Forum},
	year = {2024},	
	volume = {43},
  doi={10.1111/cgf.15083}
}
```

## License

This project is licensed under the terms of the [MIT license](https://github.com/chadepl/contour-depth/blob/main/LICENSE).
