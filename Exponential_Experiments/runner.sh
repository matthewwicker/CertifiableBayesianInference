#python3 MNIST_runner.py --eps 0.10 --lam 0.25 --rob 6 --opt HMC --gpu 0 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.25 --rob 5 --opt BBB --gpu 1 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.25 --rob 5 --opt VOGN --gpu 2 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.25 --rob 5 --opt NA --gpu 3 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.25 --rob 5 --opt SWAG --gpu 4 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.25 --rob 6 --opt SWAG --gpu 4 &
#python3 MNIST_runner.py --eps 0.11 --lam 0.25 --rob 2 --opt SGD --gpu 5 &
python3 MNIST_runner.py --eps 0.11 --lam 0.25 --rob 5 --opt SGD --gpu 4 
python3 HMC_runner.py --eps 0.11 --lam 0.25 --rob 5 --opt HMC --gpu 5 