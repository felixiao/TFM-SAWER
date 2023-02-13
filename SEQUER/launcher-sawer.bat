dt=$(date '+%y%m%d_%H%M%S');

@REM python -m torch.distributed.launch --use_env \
@REM     SEQUER/main.py --model_name sawer \
@REM     --dataset amazon-movies --cuda > "./Log/SEQ-SAWER-$dt.log" 2>"./Log/SEQ-SAWER-$dt.err"


torchrun --standalone --nproc_per_node=1 \
    SEQUER/main.py --cuda \
    --model_name sawer \
    --dataset amazon-movies  > "./Log/SEQ-SAWER-$dt.log" 2>"./Log/SEQ-SAWER-$dt.err"