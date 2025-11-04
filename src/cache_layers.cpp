#include "cache_layers.hpp"
#include <algorithm>
#include <limits>
#include <unordered_map>

namespace rdma_cache {

// L1Cache实现
L1Cache::L1Cache(size_t capacity) 
    : capacity_(capacity), hits_(0), misses_(0) {
}

bool L1Cache::lookup(uint32_t token_id, L1Entry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        entry = it->second;
        entry.last_access_ts = get_timestamp_ns();
        hits_++;
        return true;
    }
    misses_++;
    return false;
}

bool L1Cache::insert(uint32_t token_id, const Token& /* token */) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (entries_.size() >= capacity_) {
        // 需要驱逐
        uint32_t victim_id = select_victim_lfu_lru();
        if (victim_id != 0) {
            entries_.erase(victim_id);
        } else {
            return false;  // 无法插入
        }
    }
    
    L1Entry entry;
    entry.token_id = token_id;
    entry.small_fields = 0;  // 简化：实际应打包token的热字段
    entry.metadata = 0;
    entry.freq = 1;
    entry.last_access_ts = get_timestamp_ns();
    
    entries_[token_id] = entry;
    return true;
}

bool L1Cache::evict(uint32_t& evicted_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.empty()) {
        return false;
    }
    evicted_id = select_victim_lfu_lru();
    if (evicted_id != 0) {
        entries_.erase(evicted_id);
        return true;
    }
    return false;
}

void L1Cache::update_freq(uint32_t token_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        it->second.freq++;
        it->second.last_access_ts = get_timestamp_ns();
    }
}

void L1Cache::age_counters() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : entries_) {
        pair.second.freq >>= 1;  // 右移一位实现老化
    }
}

uint32_t L1Cache::select_victim_lfu_lru() {
    if (entries_.empty()) {
        return 0;
    }
    
    // LFU优先，LRU作为tie-breaker
    uint32_t victim_id = 0;
    uint32_t min_freq = std::numeric_limits<uint32_t>::max();
    uint64_t oldest_ts = std::numeric_limits<uint64_t>::max();
    
    for (const auto& pair : entries_) {
        const L1Entry& entry = pair.second;
        if (entry.freq < min_freq || 
            (entry.freq == min_freq && entry.last_access_ts < oldest_ts)) {
            min_freq = entry.freq;
            oldest_ts = entry.last_access_ts;
            victim_id = pair.first;
        }
    }
    
    return victim_id;
}

void L1Cache::reset_stats() {
    hits_ = 0;
    misses_ = 0;
}

// L2Cache实现
L2Cache::L2Cache(size_t capacity_mb)
    : capacity_bytes_(capacity_mb * 1024 * 1024), hits_(0), misses_(0) {
}

bool L2Cache::lookup(uint32_t token_id, Token& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        token = it->second;
        hits_++;
        return true;
    }
    misses_++;
    return false;
}

bool L2Cache::peek(uint32_t token_id, Token& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        token = it->second;
        return true;
    }
    return false;
}

bool L2Cache::insert(uint32_t token_id, const Token& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 简化：实际需要检查容量
    // size_t token_size = token.size();
    // 这里应该检查总容量，简化实现
    
    entries_[token_id] = token;
    return true;
}

bool L2Cache::evict(uint32_t token_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.erase(token_id) > 0;
}

size_t L2Cache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& pair : entries_) {
        total += pair.second.size();
    }
    return total;
}

void L2Cache::reset_stats() {
    hits_ = 0;
    misses_ = 0;
}

// L3Cache实现
L3Cache::L3Cache(size_t capacity_mb)
    : capacity_bytes_(capacity_mb * 1024 * 1024), hits_(0), misses_(0) {
}

bool L3Cache::lookup(uint32_t token_id, Token& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        token = it->second;
        hits_++;
        return true;
    }
    misses_++;
    return false;
}

bool L3Cache::load(uint32_t token_id, Token& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        token = it->second;
        // 不计数，因为这是miss后的后台加载，不是真正的用户访问命中
        return true;
    }
    return false;
}

bool L3Cache::insert(uint32_t token_id, const Token& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_[token_id] = token;
    return true;
}

size_t L3Cache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& pair : entries_) {
        total += pair.second.size();
    }
    return total;
}

void L3Cache::reset_stats() {
    hits_ = 0;
    misses_ = 0;
}

// TokenDirectory实现
TokenDirectory::TokenDirectory() {
}

TokenDirectoryEntry* TokenDirectory::get_entry(uint32_t token_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        return &it->second;
    }
    return nullptr;
}

TokenDirectoryEntry* TokenDirectory::get_or_create_entry(uint32_t token_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end()) {
        return &it->second;
    }
    
    // 创建新条目
    TokenDirectoryEntry entry;
    entry.token_offset = token_id;  // 简化：使用token_id作为偏移
    entry.layer = CacheLayer::L3;
    entry.version = 1;
    entry.last_access_ts = get_timestamp_ns();
    
    entries_[token_id] = entry;
    return &entries_[token_id];
}

bool TokenDirectory::update_version(uint32_t token_id, uint32_t old_version, 
                                   uint32_t new_version) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(token_id);
    if (it != entries_.end() && it->second.version == old_version) {
        it->second.version = new_version;
        return true;
    }
    return false;
}

} // namespace rdma_cache

