#python3 CIFAR_runner.py --eps 0.004 --rob 3 --lam 0.25 --opt NA --gpu 3 &
#python3 CIFAR_runner.py --eps 0.004 --rob 3 --lam 0.25 --opt VOGN --gpu 4 &
#python3 CIFAR_runner.py --eps 0.004 --rob 3 --lam 0.25 --opt SWAG --gpu 5 &
python3 CIFAR_runner.py --eps 0.004 --rob 0 --lam 0.25 --opt NA --gpu 3 &
python3 CIFAR_runner.py --eps 0.004 --rob 0 --lam 0.25 --opt VOGN --gpu 4 &
python3 CIFAR_runner.py --eps 0.004 --rob 0 --lam 0.25 --opt SWAG --gpu 5 &
