
### EVN

OS: Linux-3.10.0-327.el7.x86_64-x86_64-with-glibc2.10
GPU type: NVIDIA GeForce RTX 2080 Ti

The env of python is shown in ./env.yaml.

### run

```
python main.py --lr 0.002 --l2 0.05 --epochs 100 --alpha 1.6 --batch_size 128 --seed 1 --device cuda --verbose 0 --plot 1 --fillna 12.0 --stan_feature 0 --protein_list_input P02765 P04083 O00339 P58546 O75347 P04216 P02751 P83731 P00568 P78527 P04792 P57737 P42224 P27797 Q9HAT2 P30086 O14964 P10909 P17931 --use_wandb 1
```

the output of the code is shwon in ./main.log