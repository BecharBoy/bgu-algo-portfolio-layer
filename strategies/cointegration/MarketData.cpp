#include "MarketData.h"
#include <stdexcept>
#include <algorithm>

// ── Constructor ──────────────────────────────────────────────────────────────
MarketData::MarketData(int stocks, int days)
    : num_stocks(stocks), num_days(days)
{
    price_matrix = new double[num_stocks * num_days];
    std::fill(price_matrix, price_matrix + num_stocks * num_days, 0.0);
}

// ── Destructor ────────────────────────────────────────────────────────────────
MarketData::~MarketData() {
    delete[] price_matrix;
}

// ── Copy Constructor (deep copy) ──────────────────────────────────────────────
MarketData::MarketData(const MarketData& other)
    : num_stocks(other.num_stocks),
      num_days(other.num_days),
      stock_symbols(other.stock_symbols)
{
    price_matrix = new double[num_stocks * num_days];
    std::copy(other.price_matrix,
              other.price_matrix + num_stocks * num_days,
              price_matrix);
}

// ── Copy Assignment (copy-and-swap) ───────────────────────────────────────────
MarketData& MarketData::operator=(const MarketData& other) {
    if (this == &other) return *this;
    double* tmp = new double[other.num_stocks * other.num_days];
    std::copy(other.price_matrix,
              other.price_matrix + other.num_stocks * other.num_days,
              tmp);
    delete[] price_matrix;
    price_matrix  = tmp;
    num_stocks    = other.num_stocks;
    num_days      = other.num_days;
    stock_symbols = other.stock_symbols;
    return *this;
}

// ── Move Constructor ──────────────────────────────────────────────────────────
MarketData::MarketData(MarketData&& other) noexcept
    : price_matrix(other.price_matrix),
      num_stocks(other.num_stocks),
      num_days(other.num_days),
      stock_symbols(std::move(other.stock_symbols))
{
    other.price_matrix = nullptr;
    other.num_stocks   = 0;
    other.num_days     = 0;
}

// ── Move Assignment ───────────────────────────────────────────────────────────
MarketData& MarketData::operator=(MarketData&& other) noexcept {
    if (this == &other) return *this;
    delete[] price_matrix;
    price_matrix       = other.price_matrix;
    num_stocks         = other.num_stocks;
    num_days           = other.num_days;
    stock_symbols      = std::move(other.stock_symbols);
    other.price_matrix = nullptr;
    other.num_stocks   = 0;
    other.num_days     = 0;
    return *this;
}

// ── Accessors ─────────────────────────────────────────────────────────────────
void MarketData::set_price(int stock_idx, int day_idx, double price) {
    price_matrix[stock_idx * num_days + day_idx] = price;
}

double MarketData::get_price(int stock_idx, int day_idx) const {
    return price_matrix[stock_idx * num_days + day_idx];
}

int MarketData::get_num_stocks() const { return num_stocks; }
int MarketData::get_num_days()   const { return num_days; }

const double* MarketData::get_stock_data(int stock_idx) const {
    if (stock_idx < 0 || stock_idx >= num_stocks)
        throw std::out_of_range("MarketData::get_stock_data: index out of range");
    return &price_matrix[stock_idx * num_days];
}
