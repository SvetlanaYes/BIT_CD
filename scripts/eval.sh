#!/usr/bin/env bash

gpus=0

data_name=S2Looking  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=256
project_name=CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_trainval_test_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size}



data_name=CDD_256  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=256
project_name=CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_trainval_test_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size}



data_name=data_512  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=256
project_name=CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_trainval_test_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size}




data_name=LEVIR_CD_dataset  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=256
project_name=CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_trainval_test_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size}



data_name=SYSU_CD_256  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=256
project_name=CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_trainval_test_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size}


