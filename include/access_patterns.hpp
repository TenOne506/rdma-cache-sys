#pragma once

#include "cache_layers.hpp"
#include "tokens.hpp"
#include <queue>
#include <vector>
#include <thread>
#include <atomic>

namespace rdma_cache {

// 访问结果
enum class AccessResult {
    SUCCESS,
    HOT_INLINE_MISS,
    PREFETCH_ISSUED,
    EXPAND_ISSUED,
    ERROR
};

// 实际提供服务的缓存层
enum class ServedLayer : uint8_t {
    L1 = 1,
    L2 = 2,
    L3 = 3,
    NONE = 0
};

// 访问模式：Hot Inline (L1快速路径)
class HotInlineAccess {
public:
    HotInlineAccess(L1Cache* l1, TokenDirectory* dir);
    
    AccessResult process_wr(uint32_t token_id, const Token& token, 
                           void* op_args = nullptr);
    
private:
    L1Cache* l1_cache_;
    TokenDirectory* directory_;
    
    bool apply_op_inline(L1Entry& entry, void* op_args);
    AccessResult retry_inline_or_fallback(uint32_t token_id, 
                                         const Token& token, 
                                         void* op_args);
};

// 访问模式：Prefetched Batch (L2批量预取)
class PrefetchedBatchAccess {
public:
    static constexpr size_t PREFETCH_BATCH_THRESHOLD = 32;
    static constexpr size_t MAX_BATCH = 256;
    static constexpr size_t PAGE_PREFETCH_SIZE = 64 * 1024;  // 64KB
    
    PrefetchedBatchAccess(L1Cache* l1, L2Cache* l2, TokenDirectory* dir);
    ~PrefetchedBatchAccess();
    
    void schedule_prefetch(uint32_t token_id);
    void prefetch_worker();  // 在独立线程中运行
    void stop_worker();
    
    size_t queue_size() const;
    
private:
    L1Cache* l1_cache_;
    L2Cache* l2_cache_;
    TokenDirectory* directory_;
    
    std::queue<uint32_t> prefetch_queue_;
    mutable std::mutex queue_mutex_;
    std::atomic<bool> stop_worker_;
    std::thread worker_thread_;
    
    void cluster_by_page(const std::vector<uint32_t>& batch,
                        std::vector<std::vector<uint32_t>>& clusters);
    void dma_read_simulate(uint32_t page_addr, void* buf, size_t size);
    void install_to_l1_or_staging(uint32_t token_id, const Token& token_data);
};

// 访问模式：On-demand Expand (L3回退路径)
class OnDemandExpandAccess {
public:
    OnDemandExpandAccess(L1Cache* l1, L2Cache* l2, L3Cache* l3,
                        TokenDirectory* dir, PrefetchedBatchAccess* prefetch);
    
    AccessResult handle_cache_miss_expand(uint32_t token_id, 
                                         bool wait_for_completion);
    
private:
    L1Cache* l1_cache_;
    L2Cache* l2_cache_;
    L3Cache* l3_cache_;
    TokenDirectory* directory_;
    PrefetchedBatchAccess* prefetch_access_;
    
    Token read_shared_directory(uint32_t token_id);
    void create_expand_request(uint32_t token_id, uint32_t token_offset);
    void install_token_after_expand(uint32_t token_id, const Token& token_data);
};

// 统一的访问管理器
class AccessManager {
public:
    AccessManager(L1Cache* l1, L2Cache* l2, L3Cache* l3, TokenDirectory* dir);
    ~AccessManager();
    
    AccessResult access_token(uint32_t token_id, const Token& token,
                             void* op_args = nullptr);
    ServedLayer access_token_served_layer(uint32_t token_id, const Token& token,
                             void* op_args = nullptr);
    
    // 统计信息
    uint64_t get_l1_hits() const { return l1_cache_->get_hits(); }
    uint64_t get_l2_hits() const { return l2_cache_->get_hits(); }
    uint64_t get_l3_hits() const { return l3_cache_->get_hits(); }
    uint64_t get_total_accesses() const { return total_accesses_; }
    
    void reset_stats();
    
private:
    L1Cache* l1_cache_;
    L2Cache* l2_cache_;
    L3Cache* l3_cache_;
    TokenDirectory* directory_;
    
    HotInlineAccess hot_inline_;
    PrefetchedBatchAccess prefetched_batch_;
    OnDemandExpandAccess on_demand_expand_;
    
    std::atomic<uint64_t> total_accesses_;
};

} // namespace rdma_cache

