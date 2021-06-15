
.PHONY: install uninstall
all: install

install:
	echo "Installing tensorflow/privacy..."
	mkdir -p tensorflow_privacy
	cd tensorflow_privacy; git clone https://github.com/tensorflow/privacy;	\
	cd privacy; pip install -e ./

	echo "Installing DPAttack..."
	mkdir -p data
	mkdir -p logs
	mkdir -p models
	mkdir -p experiments
	pip install -e ./

uninstall:
	pip uninstall dpa
	rm -rf tensorflow_privacy

test:
	pytest ./test
