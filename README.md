# Opinions Vary? Diagnosis First!

This is a pytorch implementation of paper "Opinions Vary? Diagnosis First!". We propose a novel method to learn the diagnosis-first segmentation from the multiple labeled data. This method beats the popular majority vote by a large margin. 

![alt text](https://github.com/WuJunde/DiagnosisFirst/blob/master/diagsimacc.png)

## Preparation

The code is run on pytorch1.8.1 + cuda 10.1.

## Quick Start
#### Generate DFSim:

python val.py -net 'your_backbone' -mod val_ad -exp_name generate_dfsim -weights 'weights of diagnosis network'

#### Train Segmentation:

python train.py -net transunet -mod seg -exp_name repro_seg -base_weights 'weights of diagnosis network'

#### Segmentation Inference:

python val.py -net 'backbone' -mod set -exp_name val_seg -weights 'recorded weights'

See cfg.py for more avaliable parameters



### Todo list

- [ ] add requirement
- [x] del debug code
- [x] cls validation
- [ ] function name alignment
- [ ] del trials
- [ ] dataset preprocess tools

