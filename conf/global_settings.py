
import os
from datetime import datetime


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

GLAUCOMA_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
GLAUCOMA_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

MASK_TRAIN_MEAN = (2.654204690220496/255)
MASK_TRAIN_STD = (21.46473779720519/255)


#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
step_size = 10
i = 1
MILESTONES = []
while i * 5 <= EPOCH:
    MILESTONES.append(i* step_size)
    i += 1

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








