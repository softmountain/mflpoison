
set -euo pipefail

test_frequency=10

# for epoch in 500; do
#    for sr in  0.3 ; do
#       taskset -c 1-30 python3 train.py \
#       --alpha 5.0 \
#       --sample_rate ${sr} \
#       --learning_rate 0.05 \
#       --global_learning_rate 0.025 \
#       --num_epochs ${epoch} \
#       --fed_alg "fed_avg" \
#       --mu 0.01 \
#       --en_att \
#       --test_frequency ${test_frequency} \
#       --att_name fuse_base \
#       --hid_size 128 \
#       --monitor_labels 0,1,2,3,4,5 \
#       --batch_size 48
#    done
# done

for alpha in 5.0; do
   for fed_alg in fed_avg ; do
      for attack_prob in  1.0; do
         taskset -c 1-30 python3 train_label_flip.py \
            --alpha "$alpha" \
            --test_frequency ${test_frequency} \
            --sample_rate 0.3 \
            --learning_rate 0.05 \
            --global_learning_rate 0.025 \
            --num_epochs 300 \
            --fed_alg "$fed_alg" \
            --mu 0.01 \
            --en_att \
            --att_name fuse_base \
            --hid_size 128 \
            --monitor_labels 0,1,2,3,4,5 \
            --attack_prob ${attack_prob} \
            --attack_src_label 2 \
            --attack_dst_label 1 \
            --attack_client_suffix "-1,-2,-3" \
            --attack_dev_ratio 0
         done
   done
done


