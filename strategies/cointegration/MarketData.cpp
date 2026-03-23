
#include "MarketData.h"

MarketData::MarketData(int stocks, int days) {
    num_stocks = stocks;
    num_days = days;

    // defining a one dimension matrix that has all the data
    // number of cells is number of stocks * number of days
    price_matrix = new double[num_stocks * num_days];

    for (int i =0; i < num_stocks * num_days; i++) {
        price_matrix[i] = 0.0;
    }
}
MarketData::~MarketData() {
    delete[] price_matrix;
}

void MarketData::set_price(int stock_idx, int day_idx, double price) {
    // calculating the real index in the one dimension matrix
    int index = stock_idx * num_days + day_idx;
    price_matrix[index] = price;
}

double MarketData::get_price(int stock_idx, int day_idx) const {
    int index = stock_idx * num_days + day_idx;
    return price_matrix[index];
}

int MarketData::get_num_stocks() const {
    return num_stocks;
}

int MarketData::get_num_days() const {
    return num_days;
}
// return the address in the memory where the days of a certain stock start
const double *MarketData::get_stock_data(int stock_idx) const {
    // TODO: Add bounds checks for stock_idx and handle invalid request path.
    int start_index = stock_idx * num_days;
    return &price_matrix[start_index];
}
