# Diagnosis First Segmentation

This is a pytorch implementation of paper [Opinions Vary? Diagnosis First!](https://arxiv.org/abs/2202.06505) (MICCAI 2022) and its extention paper [Calibrate the inter-observer segmentation uncertainty via diagnosis-first principle](https://arxiv.org/abs/2208.03016). We propose a novel method to learn the diagnosis-first segmentation from the multiple labeled data. This method beats the popular majority vote by a large margin. 

<img src="https://github.com/WuJunde/DiagnosisFirst/blob/master/diagsimacc.png" alt="text" width="400"/>

## Preparation

The code is run on pytorch1.8.1 + cuda 10.1.

## Quick Start
#### Generate DFSim:

python val.py -net 'your_backbone' -mod val_ad -exp_name generate_dfsim -weights 'weights of diagnosis network'

#### Train Segmentation:

python train.py -net 'your_backbone' -mod seg -exp_name repro_seg -base_weights 'weights of diagnosis network'

#### Segmentation Inference:

python val.py -net 'backbone' -mod set -exp_name val_seg -weights 'recorded weights'

See cfg.py for more avaliable parameters



### Todo list

- [ ] add requirement
- [x] del debug code
- [x] cls validation
- [ ] function name alignment
- [x] del trials
- [x] dataset preprocess tools

### Cite

~~~
@inproceedings{wu2022opinions,
  title={Opinions Vary? Diagnosis First!},
  author={Wu, Junde and Fang, Huihui and Yang, Dalu and Wang, Zhaowei and Zhou, Wenshuo and Shang, Fangxin and Yang, Yehui and Xu, Yanwu},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022, Proceedings, Part II},
  pages={604--613},
  year={2022},
  organization={Springer}
}
~~~

and 

~~~
@article{wu2022calibrate,
  title={Calibrate the inter-observer segmentation uncertainty via diagnosis-first principle},
  author={Wu, Junde and Fang, Huihui and Xiong, Hoayi and Duan, Lixin and Tan, Mingkui and Yang, Weihua and Liu, Huiying and Xu, Yanwu},
  journal={arXiv preprint arXiv:2208.03016},
  year={2022}
}
~~~
