#!/bin/bash

models=("MLP" "FCN")
n_runs=5
n_epochs=300
folder_name="Series2Vec_and_ieeeppg"

for model in "${models[@]}"
do
    echo "Running for model: $model"
    # Linear probing with four domains
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_linearprobing_and_four_domains" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Results/Pre_Training/S2V/2025-03-29_07-38/WESAD_S2V_config.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_pretrained_model_last.pth \
        --folder_name $folder_name \
        --epochs $n_epochs \
        --runs $n_runs \
        --is_finetuning 0 \
        --seed 10

    # Finetuning with four domains
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_finetuning_and_four_domains" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Results/Pre_Training/S2V/2025-03-29_07-38/WESAD_S2V_config.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_pretrained_model_last.pth \
        --folder_name $folder_name \
        --epochs $n_epochs \
        --runs $n_runs \
        --is_finetuning 1 \
        --seed 20

    # Linear probing with one domain
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_linearprobing_and_one_domain" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Results/Pre_Training/S2V/2025-03-29_07-38/WESAD_ECGonly_S2V_config.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_ECGonly_pretrained_model_last.pth \
        --folder_name $folder_name \
        --epochs $n_epochs \
        --runs $n_runs \
        --is_finetuning 0 \
        --seed 30

    # Finetuning with one domain
    python finetuning.py \
        --model_name "${model}" \
        --description "${model}_finetuning_and_one_domain" \
        --head_configuration_file "Configs/${model}_headmodel.json" \
        --encoder_configuration_file Results/Pre_Training/S2V/2025-03-29_07-38/WESAD_ECGonly_S2V_config.json \
        --encoder_checkpoint_path Results/Pre_Training/S2V/2025-03-29_07-38/checkpoints/WESAD_ECGonly_pretrained_model_last.pth \
        --folder_name $folder_name \
        --epochs $n_epochs \
        --runs $n_runs \
        --is_finetuning 1 \
        --seed 40
done