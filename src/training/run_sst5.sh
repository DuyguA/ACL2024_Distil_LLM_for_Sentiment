BATCH_SIZE=64
VAL_BATCH_SIZE=64
EPOCHS=3
LR=2e-5
DATASET="sst-5"
TEXT_KEY="text"
NUM_LABELS=5
TEMPERATURE=5
LOSS_TYPE="CLS"
ALPHA=0.6
BETA=0.01
EMBEDS_FILE="../tensors/embeddings-llama3B-sst5.jsonl"
OUTPUT_DIR="llama3B-sst5-alpha0.6beta0.01"
TEACHER_DIM=3072




python3 -u trainer.py --batch_size=$BATCH_SIZE \
                      --val_batch_size=$VAL_BATCH_SIZE \
                      --epochs=$EPOCHS \
                      --lr=$LR \
                      --dataset=$DATASET \
                      --text_key=$TEXT_KEY \
                      --num_labels=$NUM_LABELS \
                      --teacher_dim=$TEACHER_DIM \
                      --temperature=$TEMPERATURE \
                      --loss_type=$LOSS_TYPE \
                      --alpha=$ALPHA \
                      --beta=$BETA \
                      --embeds_file=$EMBEDS_FILE \
		      --output_dir=$OUTPUT_DIR

