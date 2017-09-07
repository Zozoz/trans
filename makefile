all: train_transD test_transD test_transE train_transE
train_transD: train_transD.cpp
	g++ train_transD.cpp -o train_transD -O3 -pthread
test_transD: test_transD.cpp
	g++ test_transD.cpp -o test_transD -O3
train_transE: train_transE.cpp
	g++ train_transE.cpp -o train_transE -O3 -pthread
test_transE: test_transE.cpp
	g++ test_transE.cpp -o test_transE -O3
