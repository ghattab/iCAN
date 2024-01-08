# Project Title
### Interpretable Molecular Encodings and Representations for Machine Learning Tasks

## Table of Contents
- [Authors](https://github.com/ghattab/iCAN#authors)
- [Manuscript](https://github.com/ghattab/iCAN#manuscript)
- [Dependencies](https://github.com/ghattab/iCAN#dependencies)
- [Data](https://github.com/ghattab/iCAN#data)
- [Code](https://github.com/ghattab/iCAN#code)
- [Running the encoding for a single dataset](https://github.com/ghattab/iCAN#running-the-encoding-for-a-single-dataset)
- [Running the encoding and prediction pipeline for all datasets](https://github.com/ghattab/iCAN#running-the-encoding-and-prediction-pipeline-for-all-datasets)
- [License](https://github.com/ghattab/iCAN#license)
- [Contribution](https://github.com/ghattab/iCAN#contribution)

## Authors

- [Moritz Weckbecker](https://www.github.com/MoritzWeckbecker)
- [Aleksandar Anžel](https://github.com/AAnzel)
- [Zewen Yang](https://github.com/alwinyang91)
- [Georges Hattab](https://github.com/ghattab)

Created by the [Visualisation group](https://visualization.group/), which is part of the Centre for Artifical Intelligence in Public Health Research (ZKI-PH) of the Robert Koch-Institute.

## Manuscript
This package is created for a paper currently in peer review. 

Abstract:
 > Molecular encodings and their usage in machine learning models have demonstrated significant breakthroughs in biomedical applications, particularly in the classification of peptides and proteins.
To this end, we propose a new encoding method: Interpretable Carbon-based Array of Neighborhoods (iCAN). 
Designed to address machine learning models' need for more structured and less flexible input, it captures the neighborhoods of carbon atoms in a counting array and improves the utility of the resulting encodings for machine learning models.
The iCAN method provides interpretable molecular encodings and representations, enabling the comparison of molecular neighborhoods, identification of repeating patterns, and visualization of relevance heat maps for a given data set.
When reproducing a large biomedical peptide classification study, it outperforms its predecessor encoding.
When extended to proteins, it outperforms a lead structure-based encoding on 71% of the data sets. 
Our method offers interpretable encodings that can be applied to all organic molecules, including exotic amino acids, cyclic peptides, and larger proteins, making it highly versatile across various domains and data sets.
This work establishes a promising new direction for machine learning in peptide and protein classification in biomedicine and healthcare, potentially accelerating advances in drug discovery and disease diagnosis.

## Dependencies
The code is written in Python 3.7.4 and tested on Linux with the following libraries installed:

|Library|Version|
|---|---|
|altair|5.0.1|
|biopython|1.78|
|ipython|5.8.0|
|keras|2.11.0|
|matplotlib|3.5.3|
|networkx|2.6.3|
|numpy|1.21.6|
|openbabel|3.1.1|
|pandas|1.3.5|
|periodictable|1.6.1|
|pysmiles|1.0.2|
|scikit-learn|0.24.2|
|tensorflow|2.11.0|

## Data
The example datasets which are used to evaluate iCAN are collected from the [peptidereactor](https://doi.org/10.1093/nargab/lqab039) [repository](https://github.com/spaenigs/peptidereactor/tree/master/data). We took all d50 atasets from that repository and placed them at [Data/Original_datasets/](Data/Original_datasets/). Each dataset has a separate README file that contains the additional information of that data set. For the comparison between iCAN and its predecessor encoding ([CMANGOES](https://github.com/ghattab/CMANGOES)), we collected 12 additional datasets which can only be handled by these molecular encodings but not by the encodings in [peptidereactor](https://doi.org/10.1093/nargab/lqab039).

## Code
**Code for running encoding and prediction-related tasks**
|Script|Description|
|---|---|
|[Source/](./Source/)|contains all scripts necessary to run the tool.
|[Source/cenact.py](./Source/cenact.py)|contains the code that creates the iCAN encoding.
|[Source/encoding.py](./Source/encoding.py)|contains the code that encodes all datasets in Data folder.
|[Source/rfc_with_cv.py](./Source/rfc_with_cv.py)|contains the code that does training and prediction based on the encoded datasets using Random Forest Classifiers with Cross-Validation splits.
|[Source/benchmark.py](./Source/benchmark.py)|contains the code that benchmarks the runtime of the algorithm.
|[Source/cnn.py](./Code/Machine_Learning.Rmd)|contains the code that does training and prediction based on the encoded datasets using a basic Convolutional Neural Network.

**Code for creating visualisations**
|Script|Description|
|---|---|
|[Visualisation/](./Visualisation/)|contains all scripts necessary to create visualisations as well as the visualisations themselves.
|[Visualisation/mann-whitney.py](./Visualisation/mann-whitney.py)|contains the code that calculates whether the differences in prediction F1-scores for iCAN and CMANGOES are significant using a [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test).
|[Visualisation/donut.py](./Visualisation/donut.py)|contains the code to visualise the previously calculated signifcance results using a donut chart.
|[Visualisation/dumbbell-CMANGOES.py](./Visualisation/dumbbell-CMANGOES.py)|contains the code that creates a dumbbell chart comparing F1-scores for iCAN and CMANGOES encodings.
|[Visualisation/dumbbell-cnn.py](./Visualisation/dumbbell-cnn.py)|contains the code that creates a dumbbell chart comparing F1-scores for iCAN encoding calculated with a Random Forest Classifier and a Convolutional Neural Network.
|[Visualisation/benchmark.ipynb](./Visualisation/benchmark.ipynb)|contains the code that visualises the runtime of the encoding in relation to the original file size using a scatterplot.

## Running the encoding for a single dataset
Place yourself in the [Source](./Source) directory, then run the following command `python cenact.py --help` to see how to run the tool for a single dataset. The output of this option is presented here:

```
usage: CENACT [-h]
              [--alphabet_mode {without_hydrogen,with_hydrogen,data_driven}]
              [--level LEVEL] [--image {0,1}] [--show_graph SHOW_GRAPH]
              [--output_path OUTPUT_PATH]
              input_file
              
CENACT - Carbon-based Encoding of Neighbourhoods with Atom Count Tables

positional arguments:
  input_file            A required path-like argument

optional arguments:
  -h, --help            show this help message and exit
  --alphabet_mode {without_hydrogen,with_hydrogen,data_driven}
                        An optional string argument that specifies which
                        alphabet of elements the algorithm should use:
                        Possible options are only using the most abundant
                        elements in proteins and excluding hydrogen, i.e. C,
                        N, O, S ('without_hydrogen'); using the most abundant
                        elements including hydrogen, i.e. H, C, N, O, S
                        ('with_hydrogen'); and using all elements which appear
                        in the smiles strings of the dataset ('data_driven').
  --level LEVEL         An optional integer argument that specifies the upper
                        boundary of levels that should be considered. Default:
                        2 (levels 1 and 2). Any integer returns neighbourhoods
                        up to that level.
  --image {0,1}         An optional integer argument that specifies whether
                        images should be created or not. Default: 0 (without
                        images).
  --show_graph SHOW_GRAPH
                        An optional integer argument that specifies whether a
                        graph representation should be created or not.
                        Default: 0 (without representation). The user should
                        provide the number between 1 and the number of
                        sequences in the parsed input file. Example: if number
                        5 is parsed for this option, a graph representation of
                        the 5th sequence of the input file shall be created
                        and placed in the corresponding images folder.
  --output_path OUTPUT_PATH
                        An optional path-like argument. For parsed paths, the
                        directory must exist beforehand. Default:
                        ./CENACT_Encodings
```

## Running the encoding and prediction pipeline for all datasets
To encode the datasets using iCAN, run the [Source/encoding.py](./Source/encoding.py) script. To get prediction scores using Random Forest Classifiers, run the [Source/rfc_with_cv.py](./Source/rfc_with_cv.py) script afterwards.

You can add your own datasets to the pipeline by adding a folder under [Data/Original_datasets](./Data/original_datasets/) which includes:

1. The sequences of the compounds for which a property should be predicted. Sequences should be saved in a file called seqs.fasta if their sequences are provided in FASTA format or seqs.smiles if their sequences are provided in SMILES format.
2. The prediction array that contains for each compound either a 1 if the compound has the desired property, or a 0 if it lacks the desired property. The array should be saved in a filed called classes.txt with one entry per line.

## License

Licensed under [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](./LICENSE).

## Contribution

Any contribution intentionally submitted for inclusion in the work by you, shall be licensed under Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
