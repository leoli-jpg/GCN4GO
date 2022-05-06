# GCN4GO
GCN for GO term name generation. Implemented by pytorch.
## Require

Python 3.6
Pytorch >= 1.4.0


## Prepare Data

1. Unzip `data.zip` to the working directory.
Open the folder `/data/`, the files are organized in the form of the following figure:

The dataset we constructed is provided in the folder `/data/processed_data/`. There are three types of `.json` files in total:

`all_geneDe*.json` Organize data in {geneID1 : \[Alias1, Descipition1\], geneID2 : \[Alias2, Descipition2\]...} format.

`idName*.json` Organize data in {termID1 : termName1, termID2 : termName2...} format.

`shuffle_Onto2Gene*.json` Organize data in {termID1 : \[geneID1_1, geneID1_2...\], termID2 : \[geneID2_1, geneID2_2...\]...} format.

## Example input data

1. `/data/20ng.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data/corpus/20ng.txt` contains raw text of each document, each line is for the corresponding line in `/data/20ng.txt`

3. `prepare_data.py` is an example for preparing your own data, note that '\n' is removed in your documents or sentences.

## Inductive version

An inductive version of Text GCN is [fast_text_gcn](https://github.com/yao8839836/fast_text_gcn), where test documents are not included in training process.
