dt=$(date '+%y%m%d_%H%M%S');

python -u ./FMLP/main.py --data_name=reviews_new --num_hidden_layers=4 --batch_size=512 \
--hidden_size=512 \
--max_seq_length=20 \
--data_dir=./Data/Amazon/MoviesAndTV_New/ \
--output_dir=./FMLP/output/ > ./Log/FMLP-Movie-$dt.log 2>./Log/FMLP-Movie-$dt.err