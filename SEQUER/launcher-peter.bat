dt=$(date '+%y%m%d_%H%M%S');

@REM python -m torch.distributed.launch --use_env \
@REM     SEQUER/main.py --model_name sawer \
@REM     --dataset amazon-movies --cuda > "./Log/SEQ-PETER-$dt.log" 2>"./Log/SEQ-PETER-$dt.err"


torchrun \
    SEQUER/main.py --model_name peter \
    --dataset amazon-movies --cuda > "./Log/SEQ-PETER-$dt.log" 2>"./Log/SEQ-PETER-$dt.err"