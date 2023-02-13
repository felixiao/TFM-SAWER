dt=$(date '+%y%m%d_%H%M%S');

torchrun --standalone --nproc_per_node=1 \
    SEQUER/main.py --cuda \
    --model_name peter \
    --dataset amazon-movies > "./Log/SEQ-PETER-$dt.log" 2>"./Log/SEQ-PETER-$dt.err"