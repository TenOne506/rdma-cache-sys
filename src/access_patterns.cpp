#include "access_patterns.hpp"
#include <chrono>
#include <algorithm>
#include <thread>
#include <unordered_map>
#include <cstdlib>

namespace rdma_cache {

// HotInlineAccess实现
HotInlineAccess::HotInlineAccess(L1Cache* l1, TokenDirectory* dir)
    : l1_cache_(l1), directory_(dir) {
}

AccessResult HotInlineAccess::process_wr(uint32_t token_id, const Token& token,
                                        void* op_args) {
    L1Entry entry;
    if (l1_cache_->lookup(token_id, entry)) {
        // L1命中：执行inline操作
        if (apply_op_inline(entry, op_args)) {
            l1_cache_->update_freq(token_id);
            
            // 更新目录
            auto* dir_entry = directory_->get_entry(token_id);
            if (dir_entry) {
                dir_entry->freq_counter++;
                dir_entry->last_access_ts = get_timestamp_ns();
            }
            
            return AccessResult::SUCCESS;
        } else {
            return retry_inline_or_fallback(token_id, token, op_args);
        }
    }
    
    return AccessResult::HOT_INLINE_MISS;
}

bool HotInlineAccess::apply_op_inline(L1Entry& entry, void* op_args) {
    // 简化实现：实际应执行原子CAS操作
    (void)entry;
    (void)op_args;
    return true;
}

AccessResult HotInlineAccess::retry_inline_or_fallback(uint32_t token_id,
                                                      const Token& token,
                                                      void* op_args) {
    // 简化：直接返回miss，由上层处理
    (void)token_id;
    (void)token;
    (void)op_args;
    return AccessResult::HOT_INLINE_MISS;
}

// PrefetchedBatchAccess实现
PrefetchedBatchAccess::PrefetchedBatchAccess(L1Cache* l1, L2Cache* l2, 
                                             TokenDirectory* dir)
    : l1_cache_(l1), l2_cache_(l2), directory_(dir), stop_worker_(false) {
    worker_thread_ = std::thread(&PrefetchedBatchAccess::prefetch_worker, this);
}

PrefetchedBatchAccess::~PrefetchedBatchAccess() {
    stop_worker();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void PrefetchedBatchAccess::schedule_prefetch(uint32_t token_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    prefetch_queue_.push(token_id);
}

size_t PrefetchedBatchAccess::queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return prefetch_queue_.size();
}

void PrefetchedBatchAccess::prefetch_worker() {
    while (!stop_worker_) {
        std::vector<uint32_t> batch;
        
        // 从队列中取出批量token
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            size_t count = std::min(prefetch_queue_.size(), MAX_BATCH);
            for (size_t i = 0; i < count; i++) {
                if (!prefetch_queue_.empty()) {
                    batch.push_back(prefetch_queue_.front());
                    prefetch_queue_.pop();
                }
            }
        }
        
        if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // 按页聚类
        std::vector<std::vector<uint32_t>> clusters;
        cluster_by_page(batch, clusters);
        
        // 处理每个聚类
        for (const auto& cluster : clusters) {
            // 模拟DMA读取
            void* temp_buf = malloc(PAGE_PREFETCH_SIZE);
            dma_read_simulate(cluster[0] * 4096, temp_buf, PAGE_PREFETCH_SIZE);
            
            // 解析并安装到L1
            for (uint32_t token_id : cluster) {
                Token token_data;
                if (l2_cache_->peek(token_id, token_data)) {
                    install_to_l1_or_staging(token_id, token_data);
                    
                    // 更新目录
                    auto* dir_entry = directory_->get_entry(token_id);
                    if (dir_entry) {
                        dir_entry->layer = CacheLayer::L1;
                        dir_entry->version++;
                    }
                }
            }
            
            free(temp_buf);
        }
        
        // 避免CPU占用过高
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void PrefetchedBatchAccess::cluster_by_page(
    const std::vector<uint32_t>& batch,
    std::vector<std::vector<uint32_t>>& clusters) {
    
    // 简化：按页号聚类（每页4096字节）
    std::unordered_map<uint32_t, std::vector<uint32_t>> page_map;
    for (uint32_t token_id : batch) {
        uint32_t page_id = token_id / 1024;  // 简化：每页1024个token
        page_map[page_id].push_back(token_id);
    }
    
    for (auto& pair : page_map) {
        clusters.push_back(std::move(pair.second));
    }
}

void PrefetchedBatchAccess::dma_read_simulate(uint32_t page_addr, 
                                             void* buf, size_t size) {
    // 模拟DMA读取延迟
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    (void)page_addr;
    (void)buf;
    (void)size;
}

void PrefetchedBatchAccess::install_to_l1_or_staging(uint32_t token_id, 
                                                     const Token& token_data) {
    l1_cache_->insert(token_id, token_data);
}

void PrefetchedBatchAccess::stop_worker() {
    stop_worker_ = true;
}

// OnDemandExpandAccess实现
OnDemandExpandAccess::OnDemandExpandAccess(L1Cache* l1, L2Cache* l2, L3Cache* l3,
                                           TokenDirectory* dir,
                                           PrefetchedBatchAccess* prefetch)
    : l1_cache_(l1), l2_cache_(l2), l3_cache_(l3), 
      directory_(dir), prefetch_access_(prefetch) {
}

AccessResult OnDemandExpandAccess::handle_cache_miss_expand(
    uint32_t token_id, bool wait_for_completion) {
    
    // 读取目录信息
    auto* dir_entry = directory_->get_entry(token_id);
    if (!dir_entry) {
        dir_entry = directory_->get_or_create_entry(token_id);
    }
    
    if (dir_entry->layer == CacheLayer::L3) {
        // 从L3加载到L2（这是miss后的后台加载，不算L3命中）
        Token token_data;
        if (l3_cache_->load(token_id, token_data)) {
            // 更新目录计数器
            dir_entry->freq_counter++;
            dir_entry->last_access_ts = get_timestamp_ns();
            
            if (wait_for_completion) {
                // 插入L2
                l2_cache_->insert(token_id, token_data);
                dir_entry->layer = CacheLayer::L2;
                dir_entry->version++;
                
                // 这次访问通过L2满足，应该手动增加L2命中计数
                // 但不再次查找（避免重复）
                // 为了简单，我们返回SUCCESS，由上层统一管理统计
                return AccessResult::SUCCESS;
            } else {
                create_expand_request(token_id, dir_entry->token_offset);
                return AccessResult::EXPAND_ISSUED;
            }
        }
    } else if (dir_entry->layer == CacheLayer::L2) {
        // L2存在但L1缺失：更新计数器并调度prefetch
        dir_entry->freq_counter++;
        dir_entry->last_access_ts = get_timestamp_ns();
        prefetch_access_->schedule_prefetch(token_id);
        return AccessResult::PREFETCH_ISSUED;
    }
    
    return AccessResult::ERROR;
}

Token OnDemandExpandAccess::read_shared_directory(uint32_t token_id) {
    Token token;
    (void)token_id;
    return token;  // 简化实现
}

void OnDemandExpandAccess::create_expand_request(uint32_t token_id, 
                                                uint32_t token_offset) {
    // 简化：直接触发prefetch
    prefetch_access_->schedule_prefetch(token_id);
    (void)token_offset;
}

void OnDemandExpandAccess::install_token_after_expand(uint32_t token_id,
                                                      const Token& token_data) {
    // 注意：这个函数现在不再使用，因为我们在handle_cache_miss_expand中
    // 已经处理了L2插入。如果需要在L2后立即promote到L1，可以在这里实现。
    (void)token_id;
    (void)token_data;
}

// AccessManager实现
AccessManager::AccessManager(L1Cache* l1, L2Cache* l2, L3Cache* l3,
                            TokenDirectory* dir)
    : l1_cache_(l1), l2_cache_(l2), l3_cache_(l3), directory_(dir),
      hot_inline_(l1, dir),
      prefetched_batch_(l1, l2, dir),
      on_demand_expand_(l1, l2, l3, dir, &prefetched_batch_),
      total_accesses_(0) {
}

AccessManager::~AccessManager() {
    prefetched_batch_.stop_worker();
}

AccessResult AccessManager::access_token(uint32_t token_id, const Token& token,
                                         void* op_args) {
    total_accesses_++;
    
    // 首先尝试L1 Hot Inline路径
    AccessResult result = hot_inline_.process_wr(token_id, token, op_args);
    if (result == AccessResult::SUCCESS) {
        return result;
    }
    
    // L1 miss: 尝试L2
    Token l2_token;
    if (l2_cache_->lookup(token_id, l2_token)) {
        // L2命中：更新目录计数器
        auto* dir_entry = directory_->get_entry(token_id);
        if (dir_entry) {
            dir_entry->freq_counter++;
            dir_entry->last_access_ts = get_timestamp_ns();
        }
        // 调度prefetch到L1
        prefetched_batch_.schedule_prefetch(token_id);
        return AccessResult::PREFETCH_ISSUED;
    }
    
    // L2也miss: 走On-demand Expand路径（内部加载到L2）
    return on_demand_expand_.handle_cache_miss_expand(token_id, true);
}

ServedLayer AccessManager::access_token_served_layer(uint32_t token_id, const Token& token,
                                         void* op_args) {
    (void)token; (void)op_args;
    // 尝试L1
    L1Entry entry;
    if (l1_cache_->lookup(token_id, entry)) {
        return ServedLayer::L1;
    }
    // 尝试L2
    Token l2_token;
    if (l2_cache_->lookup(token_id, l2_token)) {
        prefetched_batch_.schedule_prefetch(token_id);
        return ServedLayer::L2;
    }
    // L3 提供
    Token l3_token;
    if (l3_cache_->lookup(token_id, l3_token)) {
        l2_cache_->insert(token_id, l3_token);
        prefetched_batch_.schedule_prefetch(token_id);
        return ServedLayer::L3;
    }
    return ServedLayer::NONE;
}

void AccessManager::reset_stats() {
    l1_cache_->reset_stats();
    l2_cache_->reset_stats();
    l3_cache_->reset_stats();
    total_accesses_ = 0;
}

} // namespace rdma_cache

