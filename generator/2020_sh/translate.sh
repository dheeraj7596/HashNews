tw_dataset=Twitter
wb_dataset=Weibo
data_prefix=/data1/xiuwen/twitter/tweet2020/match-using-words/withscore

if [[ $1 =~ 'Twitter' ]]
then
    dataset=${tw_dataset}
elif [[ $1 =~ 'Weibo' ]]
then
    dataset=${wb_dataset}
else
    echo 'the model name should contain dataset name'
fi


nohup \
python -u ../translate.py \
    -model saved_models/words/$1  \
    -output prediction/${1/%pt/txt} \
    -src ${data_prefix}/partial_posts.txt \
    -conversation ${data_prefix}/partial_convs.txt \
    -score ${data_prefix}/partial_scores.txt \
    -beam_size 30 \
    -max_length 10 \
    -n_best 20 \
    -batch_size 64 \
    -gpu 1 > log/translate_${1%.pt}.log  \
&& python -u ../evaluate.py \
    -tgt ${data_prefix}/partial_tags.txt \
    -pred prediction/${1/%pt/txt}  \
    >> log/translate_${1%.pt}.log &
