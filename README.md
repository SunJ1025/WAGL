# WAGL: Extreme Weather Adaptive Method for Robust and Generalizable UAV-based Cross-View Geo-localization

![Python 3.6+](https://img.shields.io/badge/README-ENGLISH-green.svg)

## [UAVM 2024](https://www.zdzheng.xyz/ACMMM2024Workshop-UAV/)

UAVs in Multimedia: Capturing the World from a New Perspective. 
This repository is the code for our paper [WAGL: Extreme Weather Adaptive Method for Robust and Generalizable UAV-based Cross-View Geo-localization](https://dl.acm.org/doi/10.1145/3689095.3689100), Thank you for your kindly attention.

## requirement
1. Download the [University-1652-WX](https://www.zdzheng.xyz/ACMMM2024Workshop-UAV) dataset
2. Prepare Data Folder 
```
├── University-1652/
│   ├── readme.txt
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
|           ├── 0002
|           ...
│       ├── street/                  /* street-view training images 
│       ├── satellite/               /* satellite-view training images       
│       ├── google/                  /* noisy street-view training images (collected from Google Image)
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_street/  
│       ├── gallery_street/ 
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
│       ├── 4K_drone/
```

## Evaluation and Get the Results in Our Paper
You can download the trained embedding files (.mat)from the following link.

### Download the trained files
[Google Driver](https://drive.google.com/drive/folders/1Kp5Aa64B9FL-cZwJO_b3zSKjIVT7e8L2?usp=sharing)



## Train and Test
We provide scripts to complete TriSSA training and testing
* Change the **data_dir** and **test_dir** paths and then run:
```shell
python train.py --gpu_ids 0 --name traied_model_name --train_all --batchsize 32  --data_dir your_data_path
```

```shell
python test.py --gpu_ids 0 --name traied_model_name --test_dir your_data_path  --batchsize 32 --which_epoch 120
```

Or simplely just try to run
```shell
python run_commond.py
```

The subbmit files for UAVM are in the dictionary acmm_files
```shell
python acmm2024_subbmit.py  # generate txt file for subbmit
```

```shell
python post_process.py  # ensemble different models 
```

## Thanks
1. Zhedong Zheng, [University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization](https://dl.acm.org/doi/10.1145/3394171.3413896)
2. Xuanmeng Zhang, [Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective](https://arxiv.org/abs/2012.07620)


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex

@inproceedings{sun2024wagl,
title={WAGL: Extreme Weather Adaptive Method for Robust and Generalizable UAV-based Cross-View Geo-localization},
author={Sun, Jian and Jiang, Xinyu and Xu, Xin and Vong, Chi-Man},
booktitle={Proceedings of the 2nd Workshop on UAVs in Multimedia: Capturing the World from a New Perspective},
pages={14--18},
year={2024}
}

@inproceedings{zheng2020university,
title={University-1652: A multi-view multi-source benchmark for drone-based geo-localization},
author={Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
booktitle={Proceedings of the 28th ACM international conference on Multimedia},
pages={1395--1403},
year={2020}
}

@article{wang2024multiple,
title={Multiple-environment Self-adaptive Network for Aerial-view Geo-localization},
author={Wang, Tingyu and Zheng, Zhedong and Sun, Yaoqi and Yan, Chenggang and Yang, Yi and Chua, Tat-Seng},
journal={Pattern Recognition},
volume={152},
pages={110363},
year={2024},
publisher={Elsevier}
}
```

<!-- ### Citation
```bibtex
@article{zhang2020understanding,
  title={Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective},
  author={Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang},
  journal={arXiv preprint arXiv:2012.07620},
  year={2020}
}
``` -->