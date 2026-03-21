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

    std::vector<PairResult> top_pairs;
    {
    py::gil_scoped_release release;

    PairScanner scanner;
    top_pairs = scanner.scan_all_pairs(market_data, num_threads, min_correlation);

    }
    std::vector<CointegratedPair> final_results;

    for (const auto& pair : top_pairs) {
        const double* stock_x = market_data.get_stock_data(pair.stock_a_idx);
        const double* stock_y = market_data.get_stock_data(pair.stock_b_idx);

        OLSResult old = MathStats::calculate_OLS(stock_x, stock_y, num_days, ols.alpha, ols.beta)
        Eigen::VectorXd spread = MathStats::calculate_spread

PYBIND11_MODULE(cointegration_engine, m) {
    // TODO: Bind the CointegratedPair struct to Python
    // TODO: Bind the run_cpp_scan function to Python
    // TODO: Add module-level docs and argument names for Python usability.
}
