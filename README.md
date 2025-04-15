# Modelling Multimodal Integration in Human Concept Processing with Vision-Language Models

This repo contains the code to replicate results from the paper _Modelling Multimodal Integration in Human Concept Processing with Vision-Language Models_, by Anna Bavaresco, Marianne de Heer Kloots, Sandro Pezzelle, and Raquel Fernández. 

The Python scripts used to extract model representations are provided within `representation_extraction`.

Jupyter notebooks to replicate results from representational similarity analysis (RSA) and ablation studies are included in `rsa`. More specifically, see `rsa/rsa.ipynb` for code to compute RSA results for the main experiments and the ablation study where only concept-words were passed to vision-language models; see `rsa/partial_correlations.ipynb` for code about the ablation study where we regressed language-only models' RDMs (representational dissimilarity matrices) out of vision-language models' RDMs.


The model-derived RDMs are publicly available at [https://zenodo.org/records/15221180](https://zenodo.org/records/15221180), while the brain-derived RDMs can be recreated using the code provided in `rsa/getting_brain_rdms.ipynb`.