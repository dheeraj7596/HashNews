tw_dataset=Twitter
wb_dataset=Weibo

python -u ../evaluate.py \
    -tgt  /home/xiuwen/true_2020.txt \
    -pred /home/xiuwen/pred.txt \
    >> log/translate_news_20_new.log &
