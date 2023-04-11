# Dynamic portfolio rebalancing through reinforcement learning 

## by Eddy Lim, Qi Cao, Chai Quek

> These are the source codes for our paper titled "Dynamic Portfolio Rebalancing through Reinforcement Learning" at Neural Computing and Application in 2022.
>
> Welcome to cite our paper. 
>
> How to cite our paper: 
>
> **Q. Y. E. Lim, Q. Cao, and C. Quek, "Dynamic portfolio rebalancing through reinforcement learning," Neural Computing and Applications, vol. 34, no. 9, pp. 7125-7139, 2022. (https://doi.org/10.1007/s00521-021-06853-3).** 
>


## Arguments
- choose_set_num: index of data_set where data_set = ['portfolio1', 'portfolio2', 'portfolio3']
- stocks: Stock tickers in order of risk (Desc)
- path: path to save, replace '/' with ','
- load: whether to load / train
- full_swing: whether to use full_swing / gradual approach

## Example gradual approach for portfolio 1

1. train without LSTM prediction (gradual approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,lagged
```
2. load without LSTM prediction (gradual approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,lagged \
    --load
```
3. train lstm model. To run for all tickers
```bash
python3 lstm_pred.py --stock=WMT --stock_file=portfolio1
```
```bash
python3 lstm_pred.py --stock=AXP --stock_file=portfolio1
```
```bash
python3 lstm_pred.py --stock=MCD --stock_file=portfolio1
```
4. train with LSTM prediction (gradual approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,non_lagged
```
5. load with LSTM prediction (gradual approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,non_lagged \
    --load
```

## Example full_swing approach for portfolio 1

1. train without LSTM prediction (full_swing approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,fs_lagged \
    --full_swing
```
2. load without LSTM prediction (full_swing approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,fs_lagged \
    --load \
    --full_swing
```
3. train lstm model. To run for all tickers
```bash
python3 lstm_pred.py --stock=WMT --stock_file=portfolio1
```
```bash
python3 lstm_pred.py --stock=AXP --stock_file=portfolio1
```
```bash
python3 lstm_pred.py --stock=MCD --stock_file=portfolio1
```
4. train with LSTM prediction (full_swing approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,fs_non_lagged \
    --full_swing
```
5. load with LSTM prediction (full_swing approach)
```bash
python3 train_wo_predict.py \
    --choose_set_num=0 \
    --stocks=WMT,AXP,MCD \
    --path=data,rl,portfolio1,fs_non_lagged \
    --load \
    --full_swing
```

## For plotting results
Use rl_visual.py. Change arguments in py script as necessary

## Environment
Code tested in python 3.7.3 64-bit, in ubuntu 18.0.4
