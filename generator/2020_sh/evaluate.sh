tw_dataset=Twitter
wb_dataset=Weibo

python -u ../evaluate.py \
    -tgt  /data1/xiuwen/twitter/tweet2020/tweet-without-conversation/new_test_tag.txt \
    -pred /data1/xiuwen/twitter/tweet2020/tweet-without-conversation/prediction.txt \
    >> log/translate_tm.log &
