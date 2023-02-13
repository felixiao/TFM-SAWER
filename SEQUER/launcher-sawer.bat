dt=$(date '+%y%m%d_%H%M%S');

torchrun --standalone --nproc_per_node=1 \
    SEQUER/main.py --cuda \
    --model_name sawer \
    --dataset amazon-movies  > "./Log/SEQ-SAWER-$dt.log" 2>"./Log/SEQ-SAWER-$dt.err"