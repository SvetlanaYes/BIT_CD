#!/usr/bin/env bash

gpus=0

data_name=data_512_crop  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=512
project_name=CD_base_transformer_pos_s4_dd8_CDD_256_b8_lr0.01_train_val_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1
testing_mode=crop  # resize, crop, sliding_window_avg, sliding_window_gauss
window_size=256

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size} --testing_mode ${testing_mode} --window_size ${window_size}



data_name=S2Looking_crop  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=1024
project_name=CD_base_transformer_pos_s4_dd8_CDD_256_b8_lr0.01_train_val_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1
testing_mode=crop  # resize, crop, sliding_window_avg, sliding_window_gauss
window_size=256

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size} --testing_mode ${testing_mode} --window_size ${window_size}




data_name=LEVIR_CD_dataset_crop  # google_earth_pro  # data_256  # LEVIR
net_G=base_transformer_pos_s4_dd8  # _dedim8
split=test
img_size=1024
project_name=CD_base_transformer_pos_s4_dd8_CDD_256_b8_lr0.01_train_val_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
checkpoint_name=best_ckpt.pt
batch_size=1
testing_mode=crop  # resize, crop, sliding_window_avg, sliding_window_gauss
window_size=256

python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size} --testing_mode ${testing_mode} --window_size ${window_size}





# data_name=CDD_256_crop  # google_earth_pro  # data_256  # LEVIR
# net_G=base_transformer_pos_s4_dd8  # _dedim8
# split=test
# img_size=256
# project_name=CD_base_transformer_pos_s4_dd8_LEVIR_CD_dataset_b8_lr0.01_train_val_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
# checkpoint_name=best_ckpt.pt
# batch_size=1
# testing_mode=crop  # resize, crop, sliding_window_avg, sliding_window_gauss
# window_size=256

# python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size} --testing_mode ${testing_mode} --window_size ${window_size}




# data_name=SYSU_CD_256_crop  # google_earth_pro  # data_256  # LEVIR
# net_G=base_transformer_pos_s4_dd8  # _dedim8
# split=test
# img_size=256
# project_name=CD_base_transformer_pos_s4_dd8_LEVIR_CD_dataset_b8_lr0.01_train_val_200_linear  # BIT_google_earth_pro  # BIT_data_256  # BIT_LEVIR
# checkpoint_name=best_ckpt.pt
# batch_size=1
# testing_mode=crop  # resize, crop, sliding_window_avg, sliding_window_gauss
# window_size=256

# python eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --batch_size ${batch_size} --testing_mode ${testing_mode} --window_size ${window_size}


