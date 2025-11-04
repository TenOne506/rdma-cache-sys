#include "simulator.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

using namespace rdma_cache;

int main(int argc, char* argv[]) {
    ExperimentConfig config;
    
    // 解析命令行参数
    if (argc > 1) {
        config.num_tokens = std::stoul(argv[1]);
    }
    if (argc > 2) {
        config.num_accesses = std::stoul(argv[2]);
    }
    if (argc > 3) {
        int workload_type = std::stoi(argv[3]);
        config.workload_type = static_cast<WorkloadType>(workload_type);
    }
    if (argc > 4) {
        config.output_file = argv[4];
    }
    if (argc > 5) {
        config.log_file = argv[5];
    }
    
    std::cout << "RDMA Cache Simulation System" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Tokens: " << config.num_tokens << std::endl;
    std::cout << "  Accesses: " << config.num_accesses << std::endl;
    std::cout << "  Workload Type: " << static_cast<int>(config.workload_type) << std::endl;
    std::cout << "  L1 Capacity: " << config.l1_capacity << " entries" << std::endl;
    std::cout << "  L2 Capacity: " << config.l2_capacity_mb << " MB" << std::endl;
    std::cout << "  L3 Capacity: " << config.l3_capacity_mb << " MB" << std::endl;
    std::cout << "============================" << std::endl;
    
    Simulator simulator(config);
    simulator.run();
    
    // 导出结果
    PerformanceMetrics metrics = simulator.get_metrics();
    simulator.export_results(config.output_file);
    
    // 打印摘要
    std::cout << "\n=== Simulation Results ===" << std::endl;
    std::cout << "Total Accesses: " << metrics.total_accesses << std::endl;
    std::cout << "L1 Hits: " << metrics.l1_hits 
              << " (" << metrics.l1_hit_rate() * 100 << "%)" << std::endl;
    std::cout << "L2 Hits: " << metrics.l2_hits 
              << " (" << metrics.l2_hit_rate() * 100 << "%)" << std::endl;
    std::cout << "L3 Hits: " << metrics.l3_hits 
              << " (" << metrics.l3_hit_rate() * 100 << "%)" << std::endl;
    std::cout << "Misses: " << metrics.misses 
              << " (" << metrics.miss_rate() * 100 << "%)" << std::endl;
    std::cout << "Overall Hit Rate: " << metrics.overall_hit_rate() * 100 << "%" << std::endl;
    std::cout << "\nMigration Statistics:" << std::endl;
    std::cout << "  Promote L3->L2: " << metrics.promote_l3_to_l2 << std::endl;
    std::cout << "  Promote L2->L1: " << metrics.promote_l2_to_l1 << std::endl;
    std::cout << "  Demote L1->L2: " << metrics.demote_l1_to_l2 << std::endl;
    std::cout << "  Demote L2->L3: " << metrics.demote_l2_to_l3 << std::endl;
    
    return 0;
}

