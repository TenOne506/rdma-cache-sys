#pragma once

#include "cache_layers.hpp"
#include "access_patterns.hpp"
#include <thread>
#include <atomic>

namespace rdma_cache {

// Promote/Demote参数
struct MigrationParams {
    uint64_t promote_to_l1_threshold = 128;      // freq hits within window
    uint64_t promote_to_l2_threshold = 16;       // freq hits within window
    uint64_t demote_from_l1_idle_ns = 1'000'000;  // 1ms未访问则考虑demote
    uint64_t demote_from_l2_idle_ns = 10'000'000; // 10ms
    uint64_t aging_interval_ns = 1'000'000'000;   // 1秒老化一次
};

// 迁移管理器
class MigrationManager {
public:
    MigrationManager(L1Cache* l1, L2Cache* l2, L3Cache* l3,
                    TokenDirectory* dir, AccessManager* access_mgr);
    ~MigrationManager();
    
    void start_maintenance_worker();
    void stop_maintenance_worker();
    
    void set_params(const MigrationParams& params) { params_ = params; }
    const MigrationParams& get_params() const { return params_; }
    
    // 统计信息
    uint64_t get_promote_l3_to_l2_count() const { return promote_l3_to_l2_; }
    uint64_t get_promote_l2_to_l1_count() const { return promote_l2_to_l1_; }
    uint64_t get_demote_l1_to_l2_count() const { return demote_l1_to_l2_; }
    uint64_t get_demote_l2_to_l3_count() const { return demote_l2_to_l3_; }
    
    void reset_stats();
    
    // 公开迁移接口供外部调用
    bool migrate_l3_to_l2(uint32_t token_id);
    bool migrate_l2_to_l1(uint32_t token_id);
    
private:
    L1Cache* l1_cache_;
    L2Cache* l2_cache_;
    L3Cache* l3_cache_;
    TokenDirectory* directory_;
    AccessManager* access_manager_;
    
    MigrationParams params_;
    std::atomic<bool> stop_worker_;
    std::thread maintenance_thread_;
    
    // 统计
    std::atomic<uint64_t> promote_l3_to_l2_;
    std::atomic<uint64_t> promote_l2_to_l1_;
    std::atomic<uint64_t> demote_l1_to_l2_;
    std::atomic<uint64_t> demote_l2_to_l3_;
    uint64_t last_aging_time_;
    
    void maintenance_worker();
    void process_maintenance();
    bool migrate_l1_to_l2(uint32_t token_id);
    bool migrate_l2_to_l3(uint32_t token_id);
};

} // namespace rdma_cache

