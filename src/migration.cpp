#include "migration.hpp"
#include <chrono>
#include <thread>

namespace rdma_cache {

MigrationManager::MigrationManager(L1Cache* l1, L2Cache* l2, L3Cache* l3,
                                  TokenDirectory* dir, AccessManager* access_mgr)
    : l1_cache_(l1), l2_cache_(l2), l3_cache_(l3),
      directory_(dir), access_manager_(access_mgr),
      stop_worker_(false),
      promote_l3_to_l2_(0), promote_l2_to_l1_(0),
      demote_l1_to_l2_(0), demote_l2_to_l3_(0),
      last_aging_time_(get_timestamp_ns()) {
}

MigrationManager::~MigrationManager() {
    stop_maintenance_worker();
}

void MigrationManager::start_maintenance_worker() {
    stop_worker_ = false;
    maintenance_thread_ = std::thread(&MigrationManager::maintenance_worker, this);
}

void MigrationManager::stop_maintenance_worker() {
    stop_worker_ = true;
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.join();
    }
}

void MigrationManager::maintenance_worker() {
    while (!stop_worker_) {
        process_maintenance();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 每10ms检查一次
    }
}

void MigrationManager::process_maintenance() {
    uint64_t now = get_timestamp_ns();
    
    // 老化计数器
    if (now - last_aging_time_ >= params_.aging_interval_ns) {
        l1_cache_->age_counters();
        last_aging_time_ = now;
    }
    
    // 注意：实际实现中应该采样目录条目，这里简化处理
}

bool MigrationManager::migrate_l3_to_l2(uint32_t token_id) {
    auto* dir_entry = directory_->get_entry(token_id);
    if (!dir_entry || dir_entry->layer != CacheLayer::L3) {
        return false;
    }
    
    if (dir_entry->state != MigrationState::NORMAL) {
        return false;  // 正在迁移中
    }
    
    // 设置迁移状态
    dir_entry->state = MigrationState::MIGRATING_TO_L2;
    
    // 从L3读取token（不计数命中）
    Token token;
    if (!l3_cache_->load(token_id, token)) {
        dir_entry->state = MigrationState::NORMAL;
        return false;
    }
    
    // 写入L2
    if (l2_cache_->insert(token_id, token)) {
        dir_entry->layer = CacheLayer::L2;
        dir_entry->state = MigrationState::NORMAL;
        dir_entry->version++;
        promote_l3_to_l2_++;
        return true;
    }
    
    dir_entry->state = MigrationState::NORMAL;
    return false;
}

bool MigrationManager::migrate_l2_to_l1(uint32_t token_id) {
    auto* dir_entry = directory_->get_entry(token_id);
    if (!dir_entry || dir_entry->layer != CacheLayer::L2) {
        return false;
    }
    
    if (dir_entry->state != MigrationState::NORMAL) {
        return false;
    }
    
    dir_entry->state = MigrationState::MIGRATING_TO_L1;
    
    // 从L2读取（不计数命中）
    Token token;
    if (!l2_cache_->peek(token_id, token)) {
        dir_entry->state = MigrationState::NORMAL;
        return false;
    }
    
    // 如果L1满了，先驱逐一个
    if (l1_cache_->size() >= l1_cache_->capacity()) {
        uint32_t victim_id;
        if (l1_cache_->evict(victim_id)) {
            auto* victim_dir = directory_->get_entry(victim_id);
            if (victim_dir) {
                victim_dir->layer = CacheLayer::L2;
                victim_dir->version++;
            }
        }
    }
    
    // 插入L1
    if (l1_cache_->insert(token_id, token)) {
        dir_entry->layer = CacheLayer::L1;
        dir_entry->state = MigrationState::NORMAL;
        dir_entry->version++;
        promote_l2_to_l1_++;
        return true;
    }
    
    dir_entry->state = MigrationState::NORMAL;
    return false;
}

bool MigrationManager::migrate_l1_to_l2(uint32_t token_id) {
    auto* dir_entry = directory_->get_entry(token_id);
    if (!dir_entry || dir_entry->layer != CacheLayer::L1) {
        return false;
    }
    
    if (dir_entry->is_pinned()) {
        return false;  // 固定条目不迁移
    }
    
    dir_entry->state = MigrationState::MIGRATING_TO_L2;
    
    // 从L1读取
    L1Entry entry;
    if (!l1_cache_->lookup(token_id, entry)) {
        dir_entry->state = MigrationState::NORMAL;
        return false;
    }
    
    Token token;  // 简化：未从entry恢复完整token
    
    // 写回L2
    if (l2_cache_->insert(token_id, token)) {
        if (dir_entry->is_dirty()) {
            dir_entry->set_dirty(false);
        }
        
        l1_cache_->evict(token_id);
        dir_entry->layer = CacheLayer::L2;
        dir_entry->state = MigrationState::NORMAL;
        dir_entry->version++;
        demote_l1_to_l2_++;
        return true;
    }
    
    dir_entry->state = MigrationState::NORMAL;
    return false;
}

bool MigrationManager::migrate_l2_to_l3(uint32_t token_id) {
    auto* dir_entry = directory_->get_entry(token_id);
    if (!dir_entry || dir_entry->layer != CacheLayer::L2) {
        return false;
    }
    
    dir_entry->state = MigrationState::MIGRATING_TO_L3;
    
    // 从L2读取（不计数命中）
    Token token;
    if (!l2_cache_->peek(token_id, token)) {
        dir_entry->state = MigrationState::NORMAL;
        return false;
    }
    
    // 写入L3
    if (l3_cache_->insert(token_id, token)) {
        l2_cache_->evict(token_id);
        dir_entry->layer = CacheLayer::L3;
        dir_entry->state = MigrationState::NORMAL;
        dir_entry->version++;
        demote_l2_to_l3_++;
        return true;
    }
    
    dir_entry->state = MigrationState::NORMAL;
    return false;
}

void MigrationManager::reset_stats() {
    promote_l3_to_l2_ = 0;
    promote_l2_to_l1_ = 0;
    demote_l1_to_l2_ = 0;
    demote_l2_to_l3_ = 0;
}

} // namespace rdma_cache

