.PHONY: build run clean

BINDING_SRC = strategies/cointegration/binding.cpp
BINDING_OUT = cointegration_engine.so

build:
	cd strategies/cointegration && \
	python -m pybind11 --includes && \
	c++ -O2 -shared -fPIC $$(python3 -m pybind11 --includes) \
		MarketData.cpp MathStats.cpp PairScanner.cpp binding.cpp \
		-o ../../$(BINDING_OUT) \
		$$(python3-config --ldflags)
	@echo "✅ cointegration_engine compiled"

run: build
	PYTHONPATH=. python main.py

clean:
	rm -f $(BINDING_OUT)
