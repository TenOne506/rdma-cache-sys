#pragma once

#include "cache_layers.hpp"
#include "access_patterns.hpp"
#include "migration.hpp"
#include "tokens.hpp"
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <fstream>
#include <atomic>

namespace rdma_cache {

// 工作负载类型
enum class WorkloadType {
    UNIFORM,      // 均匀分布
    ZIPFIAN,      // Zipfian分布（热点）
    SEQUENTIAL,   // 顺序访问
    RANDOM_WALK   // 随机游走
};

// 实验配置
struct ExperimentConfig {
    size_t num_tokens = 10000;
    size_t num_accesses = 100000;
    WorkloadType workload_type = WorkloadType::ZIPFIAN;
    double zipfian_alpha = 1.5;  // Zipfian参数
    size_t l1_capacity = 8192;
    size_t l2_capacity_mb = 1024;
    size_t l3_capacity_mb = 4096;

    // 分层延迟（纳秒）
    uint64_t l1_latency_ns = 50;
    uint64_t l2_latency_ns = 200;
    uint64_t l3_latency_ns = 800;

    // 估算通信字节：基于控制元数据
    // 每次访问的基础控制字节（doorbell、索引查找等）
    uint32_t ctrl_bytes = 64;
    // L2 路径额外数据搬运估计（例如小块WQE/目录页），按访问计
    uint32_t l2_extra_bytes = 128;
    // L3 路径额外数据搬运估计（host往返/expand），按访问计
    uint32_t l3_extra_bytes = 256;
    
    // L3 按需解压/懒加载实验参数
    double pre_stage_ratio = 0.0;      // 0.0 - 1.0 概率将L3条目预置到L2
    int expand_window = 1;             // 合并expand窗口大小
    
    MigrationParams migration_params;
    
    std::string output_file = "results.json";
    std::string log_file = "simulation.log";
};

// 性能指标
struct PerformanceMetrics {
    uint64_t total_accesses = 0;
    uint64_t l1_hits = 0;
    uint64_t l2_hits = 0;
    uint64_t l3_hits = 0;
    uint64_t misses = 0;

    uint64_t total_latency_ns = 0;
    double avg_latency_ns = 0.0;

    // 估算通信
    uint64_t total_estimated_bytes = 0;
    double estimated_comm_bw_MBps = 0.0;
    
    // L3 实验统计
    uint64_t num_expands = 0;
    uint64_t total_expand_batch = 0;
    uint64_t pre_staged_count = 0;
    
    // 迁移操作统计
    uint64_t promote_l3_to_l2 = 0;
    uint64_t promote_l2_to_l1 = 0;
    uint64_t demote_l1_to_l2 = 0;
    uint64_t demote_l2_to_l3 = 0;
    
    double l1_hit_rate() const {
        return total_accesses > 0 ? 
            static_cast<double>(l1_hits) / total_accesses : 0.0;
    }
    
    double l2_hit_rate() const {
        return total_accesses > 0 ? 
            static_cast<double>(l2_hits) / total_accesses : 0.0;
    }
    
    double l3_hit_rate() const {
        return total_accesses > 0 ? 
            static_cast<double>(l3_hits) / total_accesses : 0.0;
    }
    
    double overall_hit_rate() const {
        return total_accesses > 0 ? 
            static_cast<double>(l1_hits + l2_hits + l3_hits) / total_accesses : 0.0;
    }
    
    double miss_rate() const {
        return 1.0 - overall_hit_rate();
    }
};

// 仿真器
class Simulator {
public:
    Simulator(const ExperimentConfig& config);
    ~Simulator();
    
    void run();
    PerformanceMetrics get_metrics() const;
    
    void generate_workload();
    void execute_accesses();
    
    // 导出结果到JSON
    void export_results(const std::string& filename);
    
private:
    ExperimentConfig config_;
    
    std::unique_ptr<L1Cache> l1_cache_;
    std::unique_ptr<L2Cache> l2_cache_;
    std::unique_ptr<L3Cache> l3_cache_;
    std::unique_ptr<TokenDirectory> directory_;
    std::unique_ptr<AccessManager> access_manager_;
    std::unique_ptr<MigrationManager> migration_manager_;
    
    // Token池
    std::vector<Token> token_pool_;
    
    // 工作负载序列
    std::vector<uint32_t> workload_;
    
    // 性能指标
    PerformanceMetrics metrics_;
    
    // 随机数生成器
    std::mt19937 rng_;
    std::discrete_distribution<> zipf_dist_;
    
    // 日志
    std::ofstream log_file_;
    
    void initialize_caches();
    void populate_initial_tokens();
    uint32_t generate_token_id(WorkloadType type);
    void log_message(const std::string& msg);
    void update_metrics();
};

} // namespace rdma_cache

