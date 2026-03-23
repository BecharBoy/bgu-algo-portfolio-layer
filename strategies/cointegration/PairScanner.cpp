
#include "PairScanner.h"
#include "MathStats.h"
#include <iostream>

void PairScanner::worker_task(
    const MarketData& data,
    int start_row,
    int end_row,
    double min_correlation,
    std::vector<PairResult>& local_results)
{
    int num_stocks = data.get_num_stocks();
    int num_days   = data.get_num_days();

    for (int i = start_row; i < end_row; ++i) {
        const double* stock_a_ptr = data.get_stock_data(i);
        for (int j = i + 1; j < num_stocks; ++j) {
            const double* stock_b_ptr = data.get_stock_data(j);
            double corr = MathStats::calculate_correlation(
                stock_a_ptr, stock_b_ptr, num_days);
            if (corr >= min_correlation) {
                local_results.push_back({i, j, corr});
            }
        }
    }
}

std::vector<PairResult> PairScanner::scan_all_pairs(
    const MarketData& data, int num_threads, double min_correlation)
{
    top_pairs.clear();

    if (num_threads <= 0) num_threads = 1;

    int num_stocks = data.get_num_stocks();
    long long total_work  = (long long)num_stocks * (num_stocks - 1) / 2;
    long long target_work = total_work / num_threads;

    std::vector<std::thread> threads;
    std::vector<std::vector<PairResult>> per_thread_results(num_threads);

    int current_start = 0;
    long long current_work = 0;
    int thread_idx = 0;

    for (int i = 0; i < num_stocks; ++i) {
        current_work += num_stocks - i - 1;
        bool last_row    = (i == num_stocks - 1);
        bool quota_met   = current_work >= target_work;
        bool not_last_th = thread_idx < num_threads - 1;

        if ((quota_met && not_last_th) || last_row) {
            int end_row = last_row ? num_stocks : i + 1;
            threads.emplace_back(
                &PairScanner::worker_task, this,
                std::ref(data), current_start, end_row,
                min_correlation, std::ref(per_thread_results[thread_idx])
            );
            current_start = end_row;
            current_work  = 0;
            ++thread_idx;
            if (last_row) break;
        }
    }

    for (auto& t : threads) t.join();

    // Merge: zero contention, single-threaded merge at the end
    for (auto& local : per_thread_results) {
        top_pairs.insert(top_pairs.end(), local.begin(), local.end());
    }

    return top_pairs;
}
