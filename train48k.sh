THEANO_FLAGS=mode=FAST_RUN,device=cuda$1,floatX=float32 python -u models/two_tier/two_tier48k.py --exp $2 --n_frames 64 --frame_size 16 --emb_size 256 --skip_conn True --dim 1024 --n_rnn 5 --rnn_type LSTM --q_levels 256 --q_type mu-law --batch_size 128 --weight_norm True --learn_h0 False --which_set $2
sleep 1; 
while true; 
do 
THEANO_FLAGS=mode=FAST_RUN,device=cuda$1,floatX=float32 python -u models/two_tier/two_tier48k.py --exp $2 --n_frames 64 --frame_size 16 --emb_size 256 --skip_conn True --dim 1024 --n_rnn 5 --rnn_type LSTM --q_levels 256 --q_type mu-law --batch_size 128 --weight_norm True --learn_h0 False --which_set $2 --resume; sleep 1; done;
