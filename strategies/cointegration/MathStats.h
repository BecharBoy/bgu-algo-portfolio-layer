//
// Created by Ksenia on 06/03/2026.
//

#ifndef MATH_STATS_H
#define MATH_STATS_H
#pragma once
#include <Eigen/Dense>

struct OLSResult {
    double alpha;
    double beta;
};

class MathStats {
public:
    static double calculate_correlation(const double* stock_a, const double* stock_b, int num_days);
    static OLSResult calculate_OLS(const double* stock_x, const double* stock_y, int num_days);
    static Eigen::VectorXd calculate_spread(const double *stock_x, const double *stock_y, int num_days, double alpha, double beta);
    static double calculate_adf_statistic(const Eigen::VectorXd& spread);
};


#endif