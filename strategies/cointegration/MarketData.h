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
    // TODO: Either remove or expose symbol mapping API if this field is needed.
    std::vector<std::string> stock_symbols;

public:
    MarketData(int stocks, int days);
    ~MarketData();
    // TODO: Add bounds-safe variants returning optional/error status.
    double get_price(int stock_idx, int days_idx) const;
    void set_price(int stock_idx, int days_idx, double price);

    int get_num_stocks() const;
    int get_num_days() const;
    // TODO: Clarify pointer ownership/lifetime guarantees for callers.
    const double* get_stock_data(int stock_idx) const;
};


#endif
