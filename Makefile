.PHONY: build run clean

BINDING_SRC = strategies/cointegration/binding.cpp
BINDING_OUT = cointegration_engine.so

# Detect Eigen path: system install or local fallback
EIGEN_PATH   ?= /usr/include/eigen3
PYBIND_INC   := $(shell python3 -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell python3-config --ldflags)

build:
	cd strategies/cointegration && \
	c++ -O2 -shared -fPIC \
		$(PYBIND_INC) \
		-I$(EIGEN_PATH) \
		MarketData.cpp MathStats.cpp PairScanner.cpp binding.cpp \
		-o ../../$(BINDING_OUT) \
		$(PYTHON_LDFLAGS)
	@echo "✅ cointegration_engine compiled"

run: build
	PYTHONPATH=. python3 main.py

clean:
	rm -f $(BINDING_OUT)

# Usage:
# make build                          (uses /usr/include/eigen3)
# make build EIGEN_PATH=~/eigen3      (custom path)
