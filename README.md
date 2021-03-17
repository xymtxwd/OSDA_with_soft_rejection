# openset-DA
Pytorch implementation of Open Set Domain Adaptation with Soft Unknown-Class Rejection. 

## Requirements
- Python 3.5+
- PyTorch 0.4
- torchvision
- scikit-learn

## Usage
Run SVHN -> MNIST
```
python train_digits_osda.py --task s2m --gpu <gpu_id>
```
Run USPS -> MNIST
```
python train_digits_osda.py --task u2m --gpu <gpu_id>
```
Run MNIST -> USPS
```
python train_digits_osda.py --task m2u --gpu <gpu_id>
```
The results should look like below:
![image](https://user-images.githubusercontent.com/12399355/111513313-aca20500-871e-11eb-88dc-3759e361efaf.png)
