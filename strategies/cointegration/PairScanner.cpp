//
// Created by Ksenia on 06/03/2026.
//

#include "PairScanner.h"
#include "MathStats.h"
#include <iostream>

void PairScanner::worker_task(const MarketData &data, int start_row, int end_row, double min_correlation) {
    int num_stocks = data.get_num_stocks();
    int num_days = data.get_num_days();

    for (int i = start_row; i < end_row; ++i) {
        const double* stock_a_ptr = data.get_stock_data(i);
        for (int j = i + 1; j < num_stocks; ++j) {
            const double* stock_b_ptr = data.get_stock_data(j);
            double correlation = MathStats::calculate_correlation(stock_a_ptr, stock_b_ptr, num_days);

            // checking the limit with mutex
            if (correlation >= min_correlation) {
                std::lock_guard<std::mutex> lock(results_mutex);
                PairResult result = {i, j, correlation};
                top_pairs.push_back(result);
            }
        }
    }
}

std::vector<PairResult> PairScanner::scan_all_pairs(const MarketData &data, int num_threads, double min_correlation) {
    top_pairs.clear();
    std::vector<std::thread> threads;
    int num_stocks = data.get_num_stocks();

    // calculation of all the work
    long long total_work = (long long)num_stocks * (num_stocks -1)/ 2;

    long long target_work = total_work / num_threads;

    int current_start = 0;
    long long current_work = 0;

    // going over the rows and separate them to processes

    for (int i = 0; i < num_stocks; ++i) {
        int work_in_row = num_stocks - i - 1;
        current_work += work_in_row;

        // if we get to the working target but we got more process to make (except the last one)
        if (current_work >= target_work && threads.size() < num_threads - 1) {
            // TODO: Revisit row split boundaries to avoid skipping/duplicating work.
            int end_row = i - 1;
            threads.push_back(std::thread(&PairScanner::worker_task, this, std::ref(data),
                current_start, end_row, min_correlation));            ;

            current_start = end_row;
            current_work = 0;
        }
    }
        // TODO: Consider replacing manual thread management with a thread pool.
        threads.push_back(std::thread(&PairScanner::worker_task, this, std::ref(data), current_start, num_stocks, min_correlation));    for (int i = 0; i < num_threads; ++i) {
        // TODO: Join using threads.size() instead of num_threads for safety.
        threads[i].join();
    }
    return top_pairs;

}
