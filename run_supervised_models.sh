# Module to train FCN, MLP and HIT, i.e., HinceptionTime (ensemble of five Inceptions) to serve as baselines
python main_supervised.py --batch_size 64 \
    --num_epochs 300 \
    --lr 0.001 \
    --wdecay 0.001 \
    --runs 5 \
    --seed 42 \
    --sequence_length 720 \
    --output_dir Results/Supervised/ \