# Visualizing Herbarium Plant Specimens 
This task utilizes a SWIN Transformer fine-tuned on the top 50 species form the Asteraceae family, based on the FGVC9 2022 dataset. After extracting feature vectors from the fine-tuned model, dimensionality reduction techniques were applied to visualize the learned representations.

Two methods were explored: PCA and t-SNE. t-SNE demonstrated clearer groupings and visual separations of species than PCA. 

Experimented with:
1. Extracting feature vectors from SWIN Finetuned on top 50 species from Asteraceae family and then applying dimensionality reduction techniques.
2. Extracting feature vecotrs from SWIN Finetuned on all species from FGVC 2022 dataset and then applying dimensionality reduction techniques.
 
## Overview
1. `kaggle22_clustering.ipynb` - notebook to get feature vectors using finetuned SWIN Transformer and then performing dimensionality reduction using PCA and t-SNE 
2. `index.html` - to view visualization using t-SNE for Top 50 most occurring species in the Asteraceae Family. Points on the graph are clickable, view the herbarium specimen associated with that specific point and compare up to two images at a time. Change the JSON file to visualize a different plot. 

## Recreating the Visualization
Using the SCC, first create a Virtual Environment. Check out [these instructions](https://www.bu.edu/tech/support/research/software-and-programming/common-languages/python/python-installs/virtualenv/#create) for creating an environment using `virtualenv` for your project on the SCC. 

After creating the Virtual Environment, activate the environment and then install the necessary packages into the virtualenv by running and running the following command:

```
pip install requirements.txt
```

To recreate the visualization you will need:
1. One of the checkpoints listed below:
- Asteraceae Family, top 50 species checkpoint:`/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/astera_50/checkpoint-1300`
- Checkpoint for all species in Kaggle 2022 dataset: `/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/checkpoint-139125`
- Or your own checkpoint you wish to use. 
2. Download the images from the FGVC Kaggle 2022 competition or:
- Access to the specimen images: `/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_images/`
- Access to the test image paths: `/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/datasets/val_astera_50.json`
4. Run the cells in the `kaggle22_clustering.ipynb` notebook, the notebook will output a JSON file containing the Plotly plot. 
5. In `index.html` in the `<script>` section, replace the JSON filepath that is being fetched with the one that was just generated. 

## Viewing the Visualization 
You will first need to be added to the `herbdl` project on the SCC. 

If you are logged in at SCC OnDemand you can view the t-SNE visualization here:
`/projectnb/herbdl/workspaces/mvoong/herbdl/clustering_viz/index.html`

Otherwise, you can view the visualization [here](https://scc-ondemand1.bu.edu/pun/sys/dashboard/files/fs//projectnb/herbdl/workspaces/mvoong/herbdl/clustering_viz/index.html).

In the plot, each point represents a herbarium specimen. You can click on any point to view the associated image. All images are stored within the `/projectnb/herbdl` project, which is how the visualization is able to access and display them. The image display operates like a stack with the ability to display up to two images.