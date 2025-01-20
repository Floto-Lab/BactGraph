python scripts/train_model.py \
    --adj-matrix data/raw/llcb_perturb_hits_adj_matrix.tsv \
    --expression-data data/raw/norm_dat_pao1.tsv \
    --embeddings-dir data/processed/embeddings \
    --config configs/config.yaml \
    --train-split 0.8 \
    --seed 42
