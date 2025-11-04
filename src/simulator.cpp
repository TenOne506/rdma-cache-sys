#include "simulator.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <fstream>

namespace rdma_cache {

Simulator::Simulator(const ExperimentConfig& config)
    : config_(config), rng_(std::random_device{}()) {
    // 初始化Zipfian分布（简化实现）
    std::vector<double> weights(config.num_tokens);
    for (size_t i = 0; i < config.num_tokens; i++) {
        weights[i] = 1.0 / std::pow(i + 1, config.zipfian_alpha);
    }
    zipf_dist_ = std::discrete_distribution<>(weights.begin(), weights.end());
    
    initialize_caches();
    populate_initial_tokens();
    
    // 打开日志文件
    log_file_.open(config.log_file, std::ios::app);
    log_message("Simulator initialized");
}

Simulator::~Simulator() {
    if (migration_manager_) {
        migration_manager_->stop_maintenance_worker();
    }
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Simulator::initialize_caches() {
    l1_cache_ = std::make_unique<L1Cache>(config_.l1_capacity);
    l2_cache_ = std::make_unique<L2Cache>(config_.l2_capacity_mb);
    l3_cache_ = std::make_unique<L3Cache>(config_.l3_capacity_mb);
    directory_ = std::make_unique<TokenDirectory>();
    
    access_manager_ = std::make_unique<AccessManager>(
        l1_cache_.get(), l2_cache_.get(), l3_cache_.get(), directory_.get());
    
    migration_manager_ = std::make_unique<MigrationManager>(
        l1_cache_.get(), l2_cache_.get(), l3_cache_.get(),
        directory_.get(), access_manager_.get());
    migration_manager_->set_params(config_.migration_params);
    migration_manager_->start_maintenance_worker();
}

void Simulator::populate_initial_tokens() {
    token_pool_.resize(config_.num_tokens);
    
    // 生成不同类型的token
    for (size_t i = 0; i < config_.num_tokens; i++) {
        TokenType type = static_cast<TokenType>(i % 4);
        Token token;
        
        switch (type) {
            case TokenType::PD:
                token.pd = PDToken(static_cast<uint16_t>(i), i);
                break;
            case TokenType::MR:
                token.mr = MRToken(static_cast<uint16_t>(i), i, 0, 12, 1024);
                break;
            case TokenType::CQ:
                token.cq = CQToken(static_cast<uint16_t>(i), i, 4096, 0, 0);
                break;
            case TokenType::QP:
                token.qp = QPToken(static_cast<uint16_t>(i), i, 0, 0, 
                                  0, 0, 0, 0);
                break;
        }
        
        token_pool_[i] = token;
        
        // 初始所有token放在L3
        l3_cache_->insert(i, token);
        
        // 创建目录条目
        auto* dir_entry = directory_->get_or_create_entry(i);
        dir_entry->layer = CacheLayer::L3;
        dir_entry->token_offset = i;
    }
    
    log_message("Initialized " + std::to_string(config_.num_tokens) + " tokens");
}

void Simulator::generate_workload() {
    workload_.clear();
    workload_.reserve(config_.num_accesses);
    
    log_message("Generating workload: " + std::to_string(config_.num_accesses) + 
                " accesses, type=" + std::to_string(static_cast<int>(config_.workload_type)));
    
    for (size_t i = 0; i < config_.num_accesses; i++) {
        uint32_t token_id = generate_token_id(config_.workload_type);
        workload_.push_back(token_id);
    }
    
    log_message("Workload generation completed");
}

uint32_t Simulator::generate_token_id(WorkloadType type) {
    switch (type) {
        case WorkloadType::UNIFORM: {
            std::uniform_int_distribution<uint32_t> dist(0, config_.num_tokens - 1);
            return dist(rng_);
        }
        case WorkloadType::ZIPFIAN: {
            return zipf_dist_(rng_);
        }
        case WorkloadType::SEQUENTIAL: {
            static uint32_t seq_counter = 0;
            return (seq_counter++) % config_.num_tokens;
        }
        case WorkloadType::RANDOM_WALK: {
            static uint32_t current = 0;
            std::uniform_int_distribution<int> step(-10, 10);
            current = (current + config_.num_tokens + step(rng_)) % config_.num_tokens;
            return current;
        }
        default:
            return 0;
    }
}

void Simulator::execute_accesses() {
    log_message("Starting access execution");

    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    int current_expand_batch = 0;
    
    for (size_t i = 0; i < workload_.size(); i++) {
        uint32_t token_id = workload_[i];
        const Token& token = token_pool_[token_id];
        
        // 按 pre_stage_ratio 预置部分 L3 到 L2
        auto* dir_entry_ps = directory_->get_entry(token_id);
        if (dir_entry_ps && dir_entry_ps->layer == CacheLayer::L3) {
            if (uni01(rng_) < config_.pre_stage_ratio) {
                Token ttmp;
                if (l3_cache_->load(token_id, ttmp)) {
                    l2_cache_->insert(token_id, ttmp);
                    dir_entry_ps->layer = CacheLayer::L2;
                    metrics_.pre_staged_count++;
                }
            }
        }
        
        ServedLayer served = access_manager_->access_token_served_layer(token_id, token);
        
        metrics_.total_accesses++;
        switch (served) {
            case ServedLayer::L1:
                metrics_.total_latency_ns += config_.l1_latency_ns;
                metrics_.total_estimated_bytes += config_.ctrl_bytes;
                break;
            case ServedLayer::L2:
                metrics_.total_latency_ns += config_.l2_latency_ns;
                metrics_.total_estimated_bytes += (uint64_t)config_.ctrl_bytes + config_.l2_extra_bytes;
                break;
            case ServedLayer::L3: {
                metrics_.total_latency_ns += config_.l3_latency_ns;
                metrics_.total_estimated_bytes += (uint64_t)config_.ctrl_bytes + config_.l3_extra_bytes;
                // 统计 expand 窗口
                current_expand_batch++;
                if (current_expand_batch >= std::max(1, config_.expand_window)) {
                    metrics_.num_expands++;
                    metrics_.total_expand_batch += current_expand_batch;
                    current_expand_batch = 0;
                }
                break;
            }
            default:
                break;
        }
        
        // 迁移检查
        auto* dir_entry = directory_->get_entry(token_id);
        if (dir_entry) {
            if (dir_entry->layer == CacheLayer::L3 && 
                dir_entry->freq_counter >= config_.migration_params.promote_to_l2_threshold) {
                migration_manager_->migrate_l3_to_l2(token_id);
            }
            if (dir_entry->layer == CacheLayer::L2 && 
                dir_entry->freq_counter >= config_.migration_params.promote_to_l1_threshold) {
                migration_manager_->migrate_l2_to_l1(token_id);
            }
        }
        
        if (i % 10000 == 0) {
            update_metrics();
            if (i % 100000 == 0) {
                log_message("Progress: " + std::to_string(i) + "/" + 
                           std::to_string(workload_.size()));
            }
        }
    }
    
    // 收尾批
    if (current_expand_batch > 0) {
        metrics_.num_expands++;
        metrics_.total_expand_batch += current_expand_batch;
    }
    
    update_metrics();
    log_message("Access execution completed");
}

void Simulator::update_metrics() {
    metrics_.l1_hits = access_manager_->get_l1_hits();
    metrics_.l2_hits = access_manager_->get_l2_hits();
    metrics_.l3_hits = access_manager_->get_l3_hits();
    // 保持metrics_.total_accesses为模拟器自身累计值
    metrics_.misses = metrics_.total_accesses - 
                     (metrics_.l1_hits + metrics_.l2_hits + metrics_.l3_hits);
    metrics_.avg_latency_ns = metrics_.total_accesses > 0
        ? static_cast<double>(metrics_.total_latency_ns) / metrics_.total_accesses
        : 0.0;
    // 估算带宽（MB/s）
    double total_sec = metrics_.total_latency_ns / 1e9;
    metrics_.estimated_comm_bw_MBps = total_sec > 0.0
        ? (metrics_.total_estimated_bytes / (1024.0 * 1024.0)) / total_sec
        : 0.0;
    
    metrics_.promote_l3_to_l2 = migration_manager_->get_promote_l3_to_l2_count();
    metrics_.promote_l2_to_l1 = migration_manager_->get_promote_l2_to_l1_count();
    metrics_.demote_l1_to_l2 = migration_manager_->get_demote_l1_to_l2_count();
    metrics_.demote_l2_to_l3 = migration_manager_->get_demote_l2_to_l3_count();
}

void Simulator::run() {
    log_message("=== Simulation Started ===");
    
    generate_workload();
    execute_accesses();
    
    update_metrics();
    
    log_message("=== Simulation Completed ===");
    log_message("L1 Hit Rate: " + std::to_string(metrics_.l1_hit_rate()));
    log_message("L2 Hit Rate: " + std::to_string(metrics_.l2_hit_rate()));
    log_message("L3 Hit Rate: " + std::to_string(metrics_.l3_hit_rate()));
    log_message("Overall Hit Rate: " + std::to_string(metrics_.overall_hit_rate()));
}

PerformanceMetrics Simulator::get_metrics() const {
    return metrics_;
}

void Simulator::export_results(const std::string& filename) {
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(6);

    const double total = static_cast<double>(metrics_.total_accesses);
    const double hits_total = static_cast<double>(metrics_.l1_hits + metrics_.l2_hits + metrics_.l3_hits);
    const double l1_share = total > 0 ? static_cast<double>(metrics_.l1_hits) / total : 0.0;
    const double l2_share = total > 0 ? static_cast<double>(metrics_.l2_hits) / total : 0.0;
    const double l3_share = total > 0 ? static_cast<double>(metrics_.l3_hits) / total : 0.0;
    const double miss_share = 1.0 - (l1_share + l2_share + l3_share);

    const double l1_hit_rate_norm = hits_total > 0 ? static_cast<double>(metrics_.l1_hits) / hits_total : 0.0;
    const double l2_hit_rate_norm = hits_total > 0 ? static_cast<double>(metrics_.l2_hits) / hits_total : 0.0;
    const double l3_hit_rate_norm = hits_total > 0 ? static_cast<double>(metrics_.l3_hits) / hits_total : 0.0;

    double avg_expand_batch = metrics_.num_expands ? (double)metrics_.total_expand_batch / metrics_.num_expands : 0.0;

    out << "{\n";
    out << "  \"total_accesses\": " << metrics_.total_accesses << ",\n";
    out << "  \"l1_hits\": " << metrics_.l1_hits << ",\n";
    out << "  \"l2_hits\": " << metrics_.l2_hits << ",\n";
    out << "  \"l3_hits\": " << metrics_.l3_hits << ",\n";
    out << "  \"misses\": " << metrics_.misses << ",\n";
    out << "  \"l1_hit_rate\": " << metrics_.l1_hit_rate() << ",\n";
    out << "  \"l2_hit_rate\": " << metrics_.l2_hit_rate() << ",\n";
    out << "  \"l3_hit_rate\": " << metrics_.l3_hit_rate() << ",\n";
    out << "  \"overall_hit_rate\": " << metrics_.overall_hit_rate() << ",\n";
    out << "  \"miss_rate\": " << metrics_.miss_rate() << ",\n";
    out << "  \"avg_latency_ns\": " << metrics_.avg_latency_ns << ",\n";
    out << "  \"total_estimated_bytes\": " << metrics_.total_estimated_bytes << ",\n";
    out << "  \"estimated_comm_bw_MBps\": " << metrics_.estimated_comm_bw_MBps << ",\n";
    out << "  \"l1_share\": " << l1_share << ",\n";
    out << "  \"l2_share\": " << l2_share << ",\n";
    out << "  \"l3_share\": " << l3_share << ",\n";
    out << "  \"miss_share\": " << miss_share << ",\n";
    out << "  \"l1_hit_rate_norm\": " << l1_hit_rate_norm << ",\n";
    out << "  \"l2_hit_rate_norm\": " << l2_hit_rate_norm << ",\n";
    out << "  \"l3_hit_rate_norm\": " << l3_hit_rate_norm << ",\n";
    out << "  \"promote_l3_to_l2\": " << metrics_.promote_l3_to_l2 << ",\n";
    out << "  \"promote_l2_to_l1\": " << metrics_.promote_l2_to_l1 << ",\n";
    out << "  \"demote_l1_to_l2\": " << metrics_.demote_l1_to_l2 << ",\n";
    out << "  \"demote_l2_to_l3\": " << metrics_.demote_l2_to_l3 << ",\n";
    out << "  \"pre_stage_ratio\": " << config_.pre_stage_ratio << ",\n";
    out << "  \"expand_window\": " << config_.expand_window << ",\n";
    out << "  \"num_expands\": " << metrics_.num_expands << ",\n";
    out << "  \"avg_expand_batch\": " << avg_expand_batch << "\n";
    out << "}\n";
    out.close();
    
    log_message("Results exported to " + filename);
}

void Simulator::log_message(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&time);
    
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << " - " << msg;
    std::string log_msg = oss.str();
    
    std::cout << log_msg << std::endl;
    if (log_file_.is_open()) {
        log_file_ << log_msg << std::endl;
    }
}

} // namespace rdma_cache

