## For reproducing results of WideResNet34-10 on the CIFAR10
### PGD adversarial training
```
python main_pgd_wideresnet34_10_Validation.py --lr 0.1 --noise-coef 0.1
```

### Attack the trained model
```
python Attack_PGD_WideResNet.py --method ifgsm
```
The method can be fgsm, ifgsm, and cw
