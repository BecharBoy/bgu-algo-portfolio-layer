#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include MarketData.h
#include PairScanner.h
#include MathStats.h
#include <stdexcept>
// TODO: Include your headers (MarketData.h, PairScanner.h, MathStats.h)
// TODO: Add any extra includes needed for validation and error handling.

namespace py = pybind11;

struct CointegratedPair {
    std::string stock_x;
    std::string stock_y;
    double correlation;
    double hedge_ratio;
    double adf_stat;
};

std::vector<CointegratedPair> run_cpp_scan(

        const std::vector<std::string>& tickers,
        const std::vector<std::vector<double>>& price_matrix,
        int num_threads,
        double min_correlation) {

    // TODO: Validate input dimensions (tickers size matches matrix rows).
    // TODO: Validate equal row lengths and minimum lookback window.
    // TODO: 1. Initialize MarketData and load price_matrix into it
    // TODO: 2. Instantiate PairScanner and run scan_all_pairs
    // TODO: 3. Iterate over results, run MathStats (OLS & ADF)
    // TODO: 4. Populate and return a vector of CointegratedPair that pass the threshold


    // validetes size and shape, flatening the two dimenstion matrix into one dimension for the cpp files,

    if (tickers.empty() ||  price_matrix.empty()){
        throw std::invalid_argument("Input data cannot be empty");
    }

    if (tickers.size() != price_matrix.size()){
        throw std::invalid_argument("Mismatch between number of tickers and price matrix rows.");
    }

    int num_stocks = tickers.size();
    int num_days = price_matrix[0].size();

    for (int i = 1; i < num_stocks; ++i){
        if (price_matrix[i].size() != num_days){
            throw std::invalid_argument("Inconsistent number of days across stocks.");
           }
    }

    MarketData market_data(num_stocks, num_days);
    for (int i = 0; i < num_stocks; ++i) {
        for (int j = 0; j < num_days; ++j){
            market_data.set_price(i, j, price_matrix[i][j])
        }
    }

    // release gil to release the python thread and then activated scan_all_pairs.
    // it allows python to be free, while cpp will now use its own threading

    std::vector<PairResult> top_pairs;
    {
    py::gil_scoped_release release;

    PairScanner scanner;
    top_pairs = scanner.scan_all_pairs(market_data, num_threads, min_correlation);

    }
    std::vector<CointegratedPair> final_results;
    // go over all the pairs that follows the min_correlation, compute ols regression, to find hedge ratio and beta,
    for (const auto& pair : top_pairs) {
        const double* stock_x = market_data.get_stock_data(pair.stock_a_idx);
        const double* stock_y = market_data.get_stock_data(pair.stock_b_idx);

        OLSResult old = MathStats::calculate_OLS(stock_x, stock_y, num_days)
        Eigen::VectorXd spread = MathStats::calculate_spread(stock_x, stock_y, num_days, ols.alpha, ols.beta);

        double adf_stat = MathStats::calculate_adf_statistics(spread);
        // run at it adf, only pairs follows stationary spread (meaning the spread between two stocks will always follow a certain mean)
        // will be counted for further examination.
        // if adf < -3.0 it means theres very little chance that the spread is random and there's a big chance it follows 
        // a cointegration pattern
        if (adf_stat < -3.0) {
            CointegratedPair result;
            result.stock_x = tickers[pair.stock_a_idx];
            result.stock_y = tickers[pair.stock_b_idx];
            result.correlation = pair.correlation;
            result.hedge_ratio = ols.beta;
            results.adf_stat = adf_stat;

            final_results.push_back(result);
        }
    }
return final_results;
}

PYBIND11_MODULE(cointegration_engine, m) {
    // TODO: Bind the CointegratedPair struct to Python
    // TODO: Bind the run_cpp_scan function to Python
    // TODO: Add module-level docs and argument names for Python usability.

    // define talking back to python, it doing it by taking cointegratedPair and shows it to python as a class object
    // it doing it by giving python read only access and not full access to the object 
    m.doc() = "C++ Cointegration Engine for StatArb";

    py::class_<CointegratedPair>(m, "CointegratedPair")
        .def_readonly("stock_x", &CointegratedPair::stock_x)
        .def_readonly("stock_y", &CointegratedPair::stock_y)
        .def_readonly("correlation", &CointegratedPair::correlation)
        .def_readonly("hedge_ratio", &CointegratedPair::hedge_ratio)
        .def_readonly("adf_stat", &CointegratedPair::adf_stat)
    // binding runn_cpp_scan to allow python run it 
        m.def("run_cpp_scan", &run_cpp_scan, "Scan for cointegrated pairs",
            py::arg("tickers"),
            py::arg("price_matrix"),
            py::arg("num_threads") = 4,
            py::arg("min_correlation") = 0.85
        );


}
