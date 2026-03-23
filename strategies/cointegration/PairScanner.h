//
// Created by Ksenia on 06/03/2026.
//

#ifndef PAIR_SCANNER_H
#define PAIR_SCANNER_H

#include "MarketData.h"
#include <vector>
#include <mutex>
#include <thread>

struct PairResult {
    int stock_a_idx;
    int stock_b_idx;
    double correlation;
};
class PairScanner {
private:
    std::vector<PairResult> top_pairs;
    std::mutex results_mutex;
    // TODO: Add cap or ranking policy for very large result sets.

    // the process that every operation will run
    void worker_task(
    const MarketData& data,
    int start_row,
    int end_row,
    double min_correlation,
    std::vector<PairResult>& local_results);


public:
    // TODO: Add deterministic ordering of output pairs for reproducibility.
    std::vector<PairResult> scan_all_pairs(const MarketData& data, int num_threads, double min_correlation);
};


#endif
