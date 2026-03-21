#include "MathStats.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

double MathStats::calculate_correlation(const double *stock_a, const double *stock_b, int num_days) {
        // TODO: Guard zero-variance vectors to avoid divide-by-zero.
        Eigen::Map<const Eigen::VectorXd> vA(stock_a, num_days);
        Eigen::Map<const Eigen::VectorXd> vB(stock_b, num_days);

        Eigen::VectorXd centeredA = vA.array() - vA.mean();
        Eigen::VectorXd centeredB = vB.array() - vB.mean();

        double correlation = centeredA.dot(centeredB) / (centeredA.norm() * centeredB.norm());
        return correlation;
}

OLSResult MathStats::calculate_OLS(const double *stock_x, const double *stock_y, int num_days) {
        // TODO: Guard zero variance in X to avoid invalid beta values.
        Eigen::Map<const Eigen::VectorXd> vX(stock_x, num_days);
        Eigen::Map<const Eigen::VectorXd> vY(stock_y, num_days);

        double mean_x = vX.mean();
        double mean_y = vY.mean();

        Eigen::VectorXd centeredX = vX.array() - mean_x;
        Eigen::VectorXd centeredY = vY.array() - mean_y;

        double var_x = centeredX.dot(centeredX);
        double covariance = centeredX.dot(centeredY);

        double beta = covariance / var_x;
        double alpha = mean_y - beta * mean_x;

        return {alpha, beta};
}

Eigen::VectorXd MathStats::calculate_spread(const double *stock_x, const double *stock_y, int num_days, double alpha, double beta) {
        Eigen::Map<const Eigen::VectorXd> X(stock_x, num_days);
        Eigen::Map<const Eigen::VectorXd> Y(stock_y, num_days);

        return (Y.array() - beta * X.array() - alpha).matrix();
}

double MathStats::calculate_adf_statistic(const Eigen::VectorXd& spread) {
        // TODO: Validate sample size assumptions against production lookback.
        int n = spread.size() - 1;
        if (n <= 2) return 0.0;

        Eigen::VectorXd delta_y = spread.tail(n) - spread.head(n);
        Eigen::VectorXd y_prev = spread.head(n);

        double mean_dy = delta_y.mean();
        double mean_yp = y_prev.mean();

        Eigen::VectorXd centered_dy = delta_y.array() - mean_dy;
        Eigen::VectorXd centered_yp = y_prev.array() - mean_yp;

        double var_yp = centered_yp.dot(centered_yp);
        if (var_yp == 0) return 0.0;

        double gamma = centered_yp.dot(centered_dy) / var_yp;
        double c = mean_dy - gamma * mean_yp;

        Eigen::VectorXd residuals = (delta_y.array() - gamma * y_prev.array() - c).matrix();
    
        double mse = residuals.squaredNorm() / (n - 2); 
        double se_gamma = std::sqrt(mse / var_yp);

        if (se_gamma == 0) return 0.0;

        return gamma / se_gamma; 
}
