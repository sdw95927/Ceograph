# Ceograph
CEll spatial Organization-based graph convolutional network to integrate cell nuclei morphologies and cell-cell spatial interactions in helping clinical dicision, outcome prediction, and model interpretation. 
![Overview](./assets/fig1.PNG)

# This repository includes
* [How to construct graph from cell spatial distribution, how to construct, train, and infer a Ceograph](./scripts/ceograph_training_and_inference.ipynb)
* [How to visualize cellular contribution to Ceograph](./scripts/visualize_cellular_contribution_to_ceograph.ipynb)
* [How to visualize morphology and cell-cell interaction contribution to Ceograph](./scripts/visualize_morphology_and_cell-cell_interaction_contribution_to_ceograph.ipynb)

# Requirements
* python 3
* torch==1.0.1.post2
* torch-cluster==1.2.4
* torch-geometric==1.0.3
* torch-scatter==1.1.2
* torch-sparse==0.2.5
* torch-vision==0.1.6.dev0
* scikit-learn==0.20.3
* networkx==2.2
* more detailed environment information can be found in __Session Info__ at the bottom of .ipynb files.
