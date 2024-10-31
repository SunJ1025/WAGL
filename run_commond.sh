


# ________________ 运行之前改本次实验名字!!!!!!!!!!!!!______________________#

 python train.py \
 --name 'gem_dino-l-392-add_weather' \
 --data_name 'University_1652' \
 --batchsize 16 \
 --lr 0.01        \
 --num_epochs 60  \
 --warm_epoch 5   \
 --droprate 0.1  \
 --stride 1       \
 --h 392          \
 --w 392          \
 --sample_num 4   \
 --margin 0.3     \
 --share          \
 --triplet        \
 --DA             \
 --balance 0.9    \
--add_weather 1


python ./acmm_uavm_files/acmm2024_subbmit.py



