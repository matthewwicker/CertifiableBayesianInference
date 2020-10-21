#python3 MNIST_runner.py --eps 0.11 --lam 0.5 --rob 2 --opt HMC --gpu 0 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.75 --rob 1 --opt BBB --gpu 1 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.75 --rob 1 --opt VOGN --gpu 2 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.75 --rob 1 --opt NA --gpu 3 &
python3 MNIST_runner.py --eps 0.11 --lam 0.75 --rob 2 --opt SWAG --gpu 4 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.75 --rob 1 --opt SWAG --gpu 3 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.5 --rob 1 --opt SGD --gpu 1 
#python3 HMC_runner.py --eps 0.11 --lam 0.5 --rob 1 --opt HMC --gpu 0 
