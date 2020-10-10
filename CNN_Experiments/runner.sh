#python3 CIFAR_runner.py --eps 0.004 --rob 3 --lam 0.25 --opt NA --gpu 0 &
#python3 CIFAR_runner.py --eps 0.004 --rob 3 --lam 0.25 --opt VOGN --gpu 1 &
#python3 CIFAR_runner.py --eps 0.004 --rob 4 --lam 0.25 --opt SWAG --gpu 2 &
python3 CIFAR_runner.py --eps 0.004 --rob 0 --lam 0.25 --opt NA --gpu 0 &
python3 CIFAR_runner.py --eps 0.004 --rob 0 --lam 0.25 --opt VOGN --gpu 1 &
python3 CIFAR_runner.py --eps 0.004 --rob 0 --lam 0.25 --opt SWAG --gpu 2 &
