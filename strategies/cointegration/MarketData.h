#ifndef MARKET_DATA_H
#define MARKET_DATA_H

#include <string>
#include <vector>
#include <algorithm>

class MarketData {
private:
    double* price_matrix;
    int num_stocks;
    int num_days;
    std::vector<std::string> stock_symbols;

public:
    // Constructor / Destructor
    MarketData(int stocks, int days);
    ~MarketData();

    // Rule of Five
    MarketData(const MarketData& other);                // copy constructor
    MarketData& operator=(const MarketData& other);     // copy assignment
    MarketData(MarketData&& other) noexcept;            // move constructor
    MarketData& operator=(MarketData&& other) noexcept; // move assignment

    double get_price(int stock_idx, int day_idx) const;
    void   set_price(int stock_idx, int day_idx, double price);

    int get_num_stocks() const;
    int get_num_days()   const;

    // Returns pointer to the start of stock_idx row. Throws std::out_of_range on bad index.
    const double* get_stock_data(int stock_idx) const;
};

#endif
