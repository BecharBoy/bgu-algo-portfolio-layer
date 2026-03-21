//
// Created by Ksenia on 06/03/2026.
//

#ifndef MARKET_DATA_H
#define MARKET_DATA_H

#include <string>
#include <vector>

class MarketData {
private:
    double* price_matrix;
    int num_stocks;
    int num_days;
    std::vector<std::string> stock_symbols;

public:
    MarketData(int stocks, int days);
    ~MarketData();
    double get_price(int stock_idx, int days_idx) const;
    void set_price(int stock_idx, int days_idx, double price);

    int get_num_stocks() const;
    int get_num_days() const;
    const double* get_stock_data(int stock_idx) const;
};


#endif