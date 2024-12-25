BATCH_SIZE=64
VAL_BATCH_SIZE=32
EPOCHS=10
LR=1e-5 
DATASET="SST-5"
TEXT_KEY="text"
NUM_LABELS=5
TEMPERATURE=5
LOSS_TYPE="logits"
ALPHA=0.7
EMBEDS_FILE="../tensors/embeddings-llama1B-sst5.jsonl"




python3 -u trainer.py --batch_size=$BATCH_SIZE --val_batch_size=$VAL_BATCH_SIZE --epochs=$EPOCHS --lr=$LR   
