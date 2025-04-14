#!/bin/bash

models=("FCN" "MLP")

for model in "${models[@]}"
do
    echo "Running for model: $model"
    # Linear probing with four domains
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_linearprobing_and_four_domains" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Configs/SSL_encoder_S2V.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_pretrained_model_last.pth \
        --epochs 300 \
        --runs 5 \
        --is_finetuning 0 \
        --seed 10

    # Finetuning with four domains
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_finetuning_and_four_domains" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Configs/SSL_encoder_S2V.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_pretrained_model_last.pth \
        --epochs 300 \
        --runs 1 \
        --is_finetuning 1 \
        --seed 20

    # Linear probing with one domain
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_linearprobing_and_one_domain" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Configs/SSL_encoder_S2V.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_ECGonly_pretrained_model_last.pth \
        --epochs 300 \
        --runs 5 \
        --is_finetuning 0 \
        --seed 30

    # Finetuning with one domain
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_finetuning_and_one_domain" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Configs/SSL_encoder_S2V.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_ECGonly_pretrained_model_last.pth \
        --epochs 300 \
        --runs 5 \
        --is_finetuning 1 \
        --seed 40
done