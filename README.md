# A user-centric approach to reliable automated flow cytometry data analysis for biomedical applications
This GitHub repository provides the necessary resources to reproduce our results from model implementation and systematic quality assurance described in our user-centric solution for flow cytometry (FCM) data analysis. We encourage users and researchers to explore the tools provided here.

## Installation
1. **Set up a Python virtual environment:**
	Ensure you have set up a virt env using Python 3.8.18.This might be especially important to import required PKL files. We used conda to manage the virt env: [conda](https://docs.conda.io/projects/conda/en/4.6.0/user-guide/tasks/manage-environments.html)
2. **Install required packages:**
   We provide all packages used in a `requirements.txt` within this repository. Install all required packages with pip:
```console
pip install -r requirements.txt
```

3. **Download necessary data:**
   The data required to run the scripts and notebooks can be downloaded from the following link: [Download Data](https://cloud.izi.fraunhofer.de/s/sir7HfeSiwESZxG). 
   Ensure you place the downloaded data in the appropriate sub directory `data/`.
   Additionally, the trained Neural Network for supervised UMAP embedding is available from the same source.
   Ensure you place the downloaded model in the appropriate subdirectory `model_development/saved_models/`.

## Usage

This repository includes a series of Jupyter notebooks that guide you through the entire process of model development, and quality assurance. 

1. **Preprocessing:**
   Start with `00_preprocessing.ipynb` to generate the data required for further analysis.

2. **Data quality assurance:**
   Next, use `01_data_quality_assurance.ipynb` to reproduce our systematic data quality assurance.

2.* **Model development:**
   Before proceeding to model construction, you might want to reproduce our model development steps:
   - Navigate to the `model_development/` subdirectory.
   - Walk through  `parameter_optimization.ipynb` to reproduce the model development process.
   - For a deeper dive into our implementation of supervised UMAP via Neural Network embedding, explore `supervised_umap_embedding.ipynb` also located in `model_development/`.

3. **Model construction:**
   Continue with `02_model_construction.ipynb`, to construct the optimized example model.

5. **Preprocess all data for model quality assurance:**
   Use `03_model_quality_assurance_preprocessing.ipynb` to preprocess all data for model quality assurance.

6. **Model quality Assurance:**
   Walk through `04_model_quality_assurance.ipynb` to reproduce our results regarding our systematic model quality assurance.

#### Compuation
 
   System: Windows
   Version: 10.0.19045
   Processor: Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz, 2304 MHz
   Memory: 32 GB
   
   The time required to evaluate the notebooks from `00_preprocessing.ipynb` to `04_model_quality_assurance.ipynb` should range between 30 minutes to 60 minutes. 

   Please note, this estimation does not include Jupyter notebooks and Python scripts in `model_development/`.
   

## License
This repository and its contents are licensed under the Creative Commons Attribution-ShareAlike 4.0 International License ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt))
