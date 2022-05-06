# GCN4GO
GCN for GO term name generation. Implemented by pytorch.
## Require

Python 3.6

Pytorch >= 1.4.0


## Prepare Data

1. Unzip `data.zip` to the working directory.
    Open the folder `/data/`, the files are organized in the form of the following figure:

    The dataset we constructed is provided in the folder `/data/processed_data/`. There are three types of `.json` files in total:

    `all_geneDe*.json` Organize data in {geneID1 : \[Alias1, Descipition1\], geneID2 : \[Alias2, Descipition2\]...}.

    `idName*.json` Organize data in {termID1 : termName1, termID2 : termName2...}.

    `shuffle_Onto2Gene*.json` Organize data in {termID1 : \[geneID1_1, geneID1_2...\], termID2 : \[geneID2_1, geneID2_2...\]...}.

2. Construct data to train model.

    To use the human dataset, using the default setting.

    To use the yeast dataset, using `--yeast`.

    To use the mix dataset, using `--mix`.

    To use the pattern, using `--abbreviation`.

    Run `python gcndata_prepare.py --abbreviation --mix` to prepare data for model input.

    Run `python generate_vocab.py --abbreviation --mix` to build vocabulary.

## Train

    Run `python gcn_attention.py --abbreviation --mix --attention` to train our full model.

## Eval

    Run `python gcn_attention_eval.py --abbreviation --mix --attention --model /model/excellent` to reproduce our best results.
