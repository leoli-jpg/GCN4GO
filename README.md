# GCN4GO
GCN for GO term name generation. Implemented by pytorch.
## Require

Python 3.6
Pytorch >= 1.4.0


## Prepare Data

1. Unzip `data.zip` to the working directory.
Open the folder named data, the files are organized in the form of the following figure:

The dataset we constructed is provided in the folder `/data/processed_data/`. There are three types of `.json` files in total:
`all_geneDe*.json` Organize data in {geneID : \[Alias, Descipition\]} format.

## Example input data

1. `/data/20ng.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data/corpus/20ng.txt` contains raw text of each document, each line is for the corresponding line in `/data/20ng.txt`

3. `prepare_data.py` is an example for preparing your own data, note that '\n' is removed in your documents or sentences.

## Inductive version

An inductive version of Text GCN is [fast_text_gcn](https://github.com/yao8839836/fast_text_gcn), where test documents are not included in training process.
