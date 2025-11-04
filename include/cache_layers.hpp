#pragma once

#include "tokens.hpp"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>
#include <vector>

namespace rdma_cache {

// 缓存层枚举
enum class CacheLayer : uint8_t {
    L1 = 1,  // NIC本地SRAM
    L2 = 2,  // CXL设备内存
    L3 = 3   // Host DRAM
};

// 迁移状态
enum class MigrationState : uint8_t {
    NORMAL = 0,
    MIGRATING_TO_L1 = 1,
    MIGRATING_TO_L2 = 2,
    MIGRATING_TO_L3 = 3
};

// 目录条目
struct TokenDirectoryEntry {
    uint32_t token_offset;   // L2或L3中的偏移/物理地址
    CacheLayer layer;         // 1=L1,2=L2,3=L3 (当前主副本层)
    uint8_t flags;            // bit flags: pinned, dirty, migrating, etc.
    uint16_t reserved;
    uint32_t version;         // 版本号用于一致性
    uint64_t freq_counter;    // 访问频率计数器
    uint64_t last_access_ts;  // 最近访问时间戳（纳秒）
    MigrationState state;     // 迁移状态
    
    TokenDirectoryEntry() 
        : token_offset(0), layer(CacheLayer::L3), flags(0), reserved(0),
          version(0), freq_counter(0), last_access_ts(0),
          state(MigrationState::NORMAL) {}
    
    bool is_pinned() const { return (flags & 0x01) != 0; }
    bool is_dirty() const { return (flags & 0x02) != 0; }
    void set_pinned(bool v) { flags = v ? (flags | 0x01) : (flags & ~0x01); }
    void set_dirty(bool v) { flags = v ? (flags | 0x02) : (flags & ~0x02); }
};

// L1缓存条目（NIC SRAM，极热）
struct L1Entry {
    uint32_t token_id;       // compact id
    uint64_t small_fields;   // packed hot fields (fits inline)
    uint32_t metadata;       // e.g., version / flags
    uint32_t freq;           // approximate frequency counter
    uint64_t last_access_ts; // for LRU/LFU hybrid
    
    L1Entry() : token_id(0), small_fields(0), metadata(0), 
                freq(0), last_access_ts(0) {}
};

// L2压缩页头（CXL内存）
struct CompressedPageHdr {
    uint32_t page_id;
    uint16_t used_slots;
    uint16_t slot_bitmap;    // or larger bitmap
    // followed by compact token byte streams
};

// L1缓存层
class L1Cache {
public:
    L1Cache(size_t capacity = 8192);  // 默认8K条目
    
    bool lookup(uint32_t token_id, L1Entry& entry);
    bool insert(uint32_t token_id, const Token& token);
    bool evict(uint32_t& evicted_id);  // 返回被驱逐的ID
    void update_freq(uint32_t token_id);
    void age_counters();  // 老化频率计数器
    
    size_t size() const { return entries_.size(); }
    size_t capacity() const { return capacity_; }
    
    // 统计信息
    uint64_t get_hits() const { return hits_; }
    uint64_t get_misses() const { return misses_; }
    void reset_stats();
    
private:
    size_t capacity_;
    std::unordered_map<uint32_t, L1Entry> entries_;
    mutable std::mutex mutex_;
    
    // 统计
    std::atomic<uint64_t> hits_;
    std::atomic<uint64_t> misses_;
    
    uint32_t select_victim_lfu_lru();  // LFU/LRU混合选择
};

// L2缓存层（CXL设备内存）
class L2Cache {
public:
    L2Cache(size_t capacity_mb = 1024);  // 默认1GB
    
    bool lookup(uint32_t token_id, Token& token);
    bool peek(uint32_t token_id, Token& token);  // 不计数读取（用于后台操作）
    bool insert(uint32_t token_id, const Token& token);
    bool evict(uint32_t token_id);
    
    size_t size() const;
    size_t capacity_bytes() const { return capacity_bytes_; }
    
    // 统计信息
    uint64_t get_hits() const { return hits_; }
    uint64_t get_misses() const { return misses_; }
    void reset_stats();
    
private:
    size_t capacity_bytes_;
    std::unordered_map<uint32_t, Token> entries_;
    mutable std::mutex mutex_;
    
    // 统计
    std::atomic<uint64_t> hits_;
    std::atomic<uint64_t> misses_;
};

// L3缓存层（Host DRAM）
class L3Cache {
public:
    L3Cache(size_t capacity_mb = 4096);  // 默认4GB
    
    bool lookup(uint32_t token_id, Token& token);
    bool load(uint32_t token_id, Token& token);  // 加载但不计数命中（用于miss后的加载）
    bool insert(uint32_t token_id, const Token& token);
    
    size_t size() const;
    size_t capacity_bytes() const { return capacity_bytes_; }
    
    // 统计信息
    uint64_t get_hits() const { return hits_; }
    uint64_t get_misses() const { return misses_; }
    void reset_stats();
    
private:
    size_t capacity_bytes_;
    std::unordered_map<uint32_t, Token> entries_;
    mutable std::mutex mutex_;
    
    // 统计
    std::atomic<uint64_t> hits_;
    std::atomic<uint64_t> misses_;
};

// Token目录（全局索引）
class TokenDirectory {
public:
    TokenDirectory();
    
    TokenDirectoryEntry* get_entry(uint32_t token_id);
    TokenDirectoryEntry* get_or_create_entry(uint32_t token_id);
    
    bool update_version(uint32_t token_id, uint32_t old_version, 
                       uint32_t new_version);
    
    size_t size() const { return entries_.size(); }
    
private:
    std::unordered_map<uint32_t, TokenDirectoryEntry> entries_;
    mutable std::mutex mutex_;
};

// 获取当前时间戳（纳秒）
inline uint64_t get_timestamp_ns() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

} // namespace rdma_cache

