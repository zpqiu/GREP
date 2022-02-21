# Description
This code is for "Graph Neural News Recommendation with User Existing and Potential Interest Modeling".

# Package Structure
- src: the source code
- conf: the configure files
- data: the dataset folder

# Prepare Data
Since the dataset is very large scale and has been publiced by the MIND, you need to download it from the MIND website.
After downloading, you will get three zip files MINDlarge_train.zip, MINDlarge_dev.zip and MINDlarge_test.zip.

Then, uncompress them and put their subfiles to the corresponding subfolders of data.
For example, the files in MINDlarge_train.zip should be put at the "data/L/train"

Afer that, the complete package structure is:
```
|-- src
|-- conf
`-- data
    |-- L
        |-- train
            |-- behaviors.tsv
            |-- news.tsv
        |-- dev
            |-- behaviors.tsv
            |-- news.tsv
        |-- test 
            |-- behaviors.tsv
            |-- news.tsv
        |-- result
        `-- hop1_cocur_bip_hist50
```

# Set Environments
We first need to create a *python=3.6* virtualenv and activate it.

Then, we should intall some dependencies.
```shell
pip install -r requirements.txt
``` 

Next, we should set the system environment $MINDWD as the current path.
For example,
```shell
export MINDWD=/home/root/KG-Recommender
```
Replace `/home/root` as the real parent directory path of `KG-Recommender`.

# Source Code

Please read the *src/README.md* to get more details about the source code and the usages.

## Citation
If you use this code, please cite the paper.
```
@article{Qiu2022GraphNN,
  title={Graph Neural News Recommendation with User Existing and Potential Interest Modeling},
  author={Zhaopeng Qiu and Yunfan Hu and Xian Wu},
  journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
  year={2022}
}
```
