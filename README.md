## Preparation
pytorch1.8.1 + cuda 10.1
## Quick Start
Generate DFSim:

python val.py -net 'your_backbone' -mod val_ad -exp_name generate_dfsim -weights 'weights of diagnosis network'

Train Segmentation:

python train.py -net transunet -mod seg -exp_name repro_seg -base_weights 'weights of diagnosis network'

See cfg.py for more avaliable parameters

