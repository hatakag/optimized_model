# My proposed model


## Prerequisites

* Computer with RAM>=35GB, 64-bit OS / Google Colab, Hardware accelerator: TPU, RAM: 35GB (Free)
* Python==3.6 & pip>=19.0

## Installation

* Download or clone the project from [here](https://github.com/hatakag/optimized_model.git)
* Change working directory (cd) to the project folder
* In this folder, open the command line and enter
```
pip install -r requirements.txt
```

## Datasets

> Please first download the datasets [here](https://drive.google.com/drive/folders/15idylZGvj0Dxm1Ey4D7K-AT4FzVgMgoK?usp=sharing) and extract them into `data/` directory.

Initial datasets are from [HGCN-JE-JR](https://github.com/StephanieWyt/HGCN-JE-JR).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links encoded by ids;
* ref_r_ids: relation links encoded by ids;
* rel_ids_1: ids for entities in source KG (ZH);
* rel_ids_2: ids for entities in target KG (EN);
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;

## Running

* Modify language or some other settings in *include/Config.py*
* cd to the directory of *main.py*
* Shuffle the alignment seeds 
```shuf data/zh_en/ref_ent_ids -o data/zh_en/shuffled_ref_ent_ids``` 
(Use this to keep the train/test seeds the same when restore model from checkpoints)
* Modify start state in *include/Config.py*
* run *main.py*

## Citations

*Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang, Rui Yan, Dongyan Zhao. Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pages 5278-5284, 2019.*

```
@inproceedings{ijcai2019-733,
  title={Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs},
  author={Wu, Yuting and Liu, Xiao and Feng, Yansong and Wang, Zheng and Yan, Rui and Zhao, Dongyan},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},            
  pages={5278--5284},
  year={2019},
}
```

```
@inproceedings{wu2019jointly,
    title = "Jointly Learning Entity and Relation Representations for Entity Alignment",
    author = "Wu, Yuting  and
      Liu, Xiao  and
      Feng, Yansong  and
      Wang, Zheng  and
      Zhao, Dongyan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1023",
    doi = "10.18653/v1/D19-1023",
    pages = "240--249",
}
```