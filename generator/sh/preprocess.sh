data_tag='Twitter'
dataset=/data1/xiuwen/twitter/tweet2018/match-using-entity/modifiedbm25

if [[ $data_tag =~ 'Twitter' ]]
then
    vs=30000
    sl=35
    slt=35
    cl=200
    clt=100
    tl=10
elif [[ $data_tag =~ 'Weibo' ]]
then
    vs=50000
    sl=100
    slt=50
    cl=200
    clt=100
    tl=10
else
    echo 'Wrong dataset name!!'
fi


if [[ ! -e ../processed_data ]]
then
    mkdir ../processed_data
fi

full_data_tag=${data_tag}_src${slt}_conv${clt}_tgt${tl}_vs${vs}_notshare


python -u ../preprocess.py \
    -max_shard_size 52428800 \
    -train_src $dataset/train_repeat_post.txt \
    -train_conv $dataset/train_repeat_conv.txt \
    -train_score $dataset/train_repeat_score.txt \
    -train_tgt $dataset/train_repeat_tag.txt \
    -valid_src $dataset/valid_repeat_post.txt \
    -valid_conv $dataset/valid_repeat_conv.txt \
    -valid_score $dataset/valid_repeat_score.txt \
    -valid_tgt $dataset/valid_repeat_tag.txt \
    -save_data ../processed_data/incorpscore/${full_data_tag}  \
    -src_vocab_size ${vs} \
    -src_seq_length ${sl} \
    -conversation_seq_length ${cl} \
    -tgt_seq_length ${tl} \
    -src_seq_length_trunc ${slt} \
    -conversation_seq_length_trunc ${clt} \
    -dynamic_dict \
    # -share_vocab


