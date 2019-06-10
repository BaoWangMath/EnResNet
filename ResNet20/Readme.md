## For reproducing results of En_5ResNet20 on the CIFAR10
### PGD adversarial training
```
python main_pgd_enresnet5_20.py --lr 0.1 --noise-coef 0.1
```

### Attack the trained model
```
python Attack_PGD_EnResNet5_20.py --method ifgsm
```
The method can be fgsm, ifgsm, and cw
