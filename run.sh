#!/bin/bash
#simple2 class_mahalanobis she ash rankfeat dice knn vim kl_matching mls react gradnorm ebo gram rmds mds_ensemble mds odin scale msp; do
#for i in simple simple2 class_mahalanobis she ash dice knn vim kl_matching mls react gradnorm ebo gram rmds mds odin scale msp rankfeat mds_ensemble; do
# gram 
for i in nodetector simple class_mahalanobis she ash dice knn vim kl_matching mls react gradnorm ebo gram rmds mds odin scale msp rankfeat mds_ensemble; do; do
  echo '---------------------------------------'
  echo $i
  python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
      --postprocessor $i \
      --save-score --save-csv

done

      # --id-data cifar10 \
      # --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \

    # --id-data cifar100 \
    # --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \

    # --id-data imagenet200 \
    # --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
  