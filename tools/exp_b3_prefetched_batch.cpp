#include "cache_layers.hpp"
#include "tokens.hpp"
#include "access_patterns.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <queue>

using namespace rdma_cache;

// 实验B3配置
struct B3Config {
    // 测试参数
    size_t batch_sizes[6] = {8, 16, 32, 64, 128, 256};  // batch size
    size_t cluster_sizes_kb[4] = {4, 16, 64, 256};  // cluster by page size (KB)
    size_t token_count = 10000;  // 总token数量
    size_t test_duration_sec = 60;  // 每个测试运行60秒
    
    // 延迟模拟（纳秒）
    uint64_t l1_latency_ns = 15;  // L1访问延迟
    uint64_t l2_latency_ns = 80;  // L2访问延迟
    uint64_t l3_latency_ns = 800;  // L3访问延迟
    uint64_t dma_setup_latency_ns = 100;  // DMA设置延迟
    double dma_bandwidth_mbps = 32000.0;  // DMA带宽（MB/s）
    
    // CPU配置
    uint32_t cpu_cores = 8;
    double cpu_frequency_ghz = 2.4;
    
    std::string output_file = "results/exp_b3_results.json";
};

// 性能指标
struct B3Metrics {
    uint64_t total_ops = 0;
    uint64_t l1_hits = 0;
    uint64_t l2_hits = 0;
    uint64_t l3_hits = 0;
    uint64_t l2_to_l1_promotions = 0;  // L2->L1成功预取次数
    uint64_t prefetch_requests = 0;  // 预取请求总数
    double avg_latency_ns = 0.0;
    double throughput_ops_per_sec = 0.0;
    double dma_bandwidth_mbps_used = 0.0;
    double cpu_usage_percent = 0.0;
    
    // 命中率
    double l1_hit_rate = 0.0;
    double l2_hit_rate = 0.0;
    double l3_hit_rate = 0.0;
    double l2_to_l1_success_rate = 0.0;  // L2->L1成功率
    
    void reset() {
        total_ops = 0;
        l1_hits = 0;
        l2_hits = 0;
        l3_hits = 0;
        l2_to_l1_promotions = 0;
        prefetch_requests = 0;
        avg_latency_ns = 0.0;
        throughput_ops_per_sec = 0.0;
        dma_bandwidth_mbps_used = 0.0;
        cpu_usage_percent = 0.0;
        l1_hit_rate = 0.0;
        l2_hit_rate = 0.0;
        l3_hit_rate = 0.0;
        l2_to_l1_success_rate = 0.0;
    }
};

// Prefetched Batch模拟器
class PrefetchedBatchSimulator {
public:
    PrefetchedBatchSimulator(const B3Config& config, size_t batch_size, size_t cluster_size_kb)
        : config_(config), batch_size_(batch_size), cluster_size_bytes_(cluster_size_kb * 1024),
          l1_cache_(8192), l2_cache_(1024), l3_cache_(4096),
          stop_worker_(false), directory_() {
        
        // 初始化所有token到L3
        for (size_t i = 0; i < config_.token_count; i++) {
            Token token;
            token.qp = QPToken(static_cast<uint16_t>(i), i, 0, 0, 0, 0, 0, 0);
            l3_cache_.insert(i, token);
            
            // 同时在目录中注册
            directory_.get_or_create_entry(i);
        }
        
        // 启动预取工作线程
        worker_thread_ = std::thread(&PrefetchedBatchSimulator::prefetch_worker, this);
    }
    
    ~PrefetchedBatchSimulator() {
        stop_worker_ = true;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    B3Metrics run_test() {
        B3Metrics metrics;
        
        std::atomic<uint64_t> completed_ops(0);
        std::atomic<uint64_t> l1_hits(0);
        std::atomic<uint64_t> l2_hits(0);
        std::atomic<uint64_t> l3_hits(0);
        std::atomic<uint64_t> l2_to_l1_promotions(0);
        std::atomic<uint64_t> prefetch_requests(0);
        std::atomic<uint64_t> total_latency_ns(0);
        
        std::vector<std::thread> threads;
        std::mutex metrics_mutex;
        
        // 设置promotion计数器指针（供prefetch_worker使用）
        promotion_counter_ = &l2_to_l1_promotions;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(config_.test_duration_sec);
        
        // 启动访问线程（模拟中等频度的访问）
        const size_t num_threads = 8;
        for (size_t t = 0; t < num_threads; t++) {
            threads.emplace_back([&, t]() {
                std::mt19937 rng(std::random_device{}() + t);
                // 使用Zipfian分布模拟中等频度的访问（扫描、短流模式）
                std::vector<double> weights(config_.token_count);
                double sum = 0.0;
                double zipf_alpha = 1.2;  // 中等偏斜度
                for (size_t i = 0; i < config_.token_count; i++) {
                    weights[i] = 1.0 / std::pow(i + 1, zipf_alpha);
                    sum += weights[i];
                }
                for (size_t i = 0; i < config_.token_count; i++) {
                    weights[i] /= sum;
                }
                std::discrete_distribution<size_t> token_dist(weights.begin(), weights.end());
                
                while (std::chrono::high_resolution_clock::now() < end_time) {
                    uint32_t token_id = static_cast<uint32_t>(token_dist(rng));
                    
                    auto op_start = std::chrono::high_resolution_clock::now();
                    
                    // 尝试L1查找
                    L1Entry l1_entry;
                    if (l1_cache_.lookup(token_id, l1_entry)) {
                        l1_hits++;
                        simulate_delay_ns(config_.l1_latency_ns);
                        total_latency_ns += config_.l1_latency_ns;
                    } else {
                        // 尝试L2查找
                        Token token;
                        if (l2_cache_.lookup(token_id, token)) {
                            l2_hits++;
                            simulate_delay_ns(config_.l2_latency_ns);
                            total_latency_ns += config_.l2_latency_ns;
                            
                            // 触发预取（L2->L1）
                            schedule_prefetch(token_id);
                            prefetch_requests++;
                        } else {
                            // L3查找
                            Token token;
                            bool l3_found = l3_cache_.lookup(token_id, token);
                            if (!l3_found) {
                                token.qp = QPToken(static_cast<uint16_t>(token_id), token_id, 0, 0, 0, 0, 0, 0);
                                l3_cache_.insert(token_id, token);
                            }
                            l3_hits++;
                            simulate_delay_ns(config_.l3_latency_ns);
                            total_latency_ns += config_.l3_latency_ns;
                            
                            // 从L3加载到L2
                            l2_cache_.insert(token_id, token);
                            
                            // 触发预取（L2->L1）
                            schedule_prefetch(token_id);
                            prefetch_requests++;
                        }
                    }
                    
                    auto op_end = std::chrono::high_resolution_clock::now();
                    uint64_t op_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        op_end - op_start).count();
                    
                    completed_ops++;
                }
            });
        }
        
        // 等待所有线程完成
        for (auto& t : threads) {
            t.join();
        }
        
        auto final_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            final_time - start_time).count();
        
        metrics.total_ops = completed_ops;
        metrics.l1_hits = l1_hits;
        metrics.l2_hits = l2_hits;
        metrics.l3_hits = l3_hits;
        metrics.l2_to_l1_promotions = l2_to_l1_promotions;
        metrics.prefetch_requests = prefetch_requests;
        
        // 计算命中率
        if (metrics.total_ops > 0) {
            metrics.l1_hit_rate = static_cast<double>(metrics.l1_hits) / metrics.total_ops;
            metrics.l2_hit_rate = static_cast<double>(metrics.l2_hits) / metrics.total_ops;
            metrics.l3_hit_rate = static_cast<double>(metrics.l3_hits) / metrics.total_ops;
        }
        
        // 计算L2->L1成功率
        if (metrics.prefetch_requests > 0) {
            metrics.l2_to_l1_success_rate = static_cast<double>(metrics.l2_to_l1_promotions) / metrics.prefetch_requests;
        }
        
        // 计算平均延迟
        if (metrics.total_ops > 0) {
            metrics.avg_latency_ns = static_cast<double>(total_latency_ns) / metrics.total_ops;
        }
        
        // 计算吞吐量
        double total_time_sec = duration / 1000.0;
        if (total_time_sec > 0) {
            metrics.throughput_ops_per_sec = metrics.total_ops / total_time_sec;
        }
        
        // 计算DMA带宽使用
        // 假设每个token平均16字节，预取时会批量读取
        size_t tokens_per_cluster = cluster_size_bytes_ / 16;
        size_t clusters_per_batch = std::max(size_t(1), batch_size_ / tokens_per_cluster);
        size_t total_bytes_prefetched = metrics.l2_to_l1_promotions * 16;  // 每个token 16字节
        double dma_time_sec = total_bytes_prefetched / (config_.dma_bandwidth_mbps * 1024.0 * 1024.0 / 8.0);
        if (total_time_sec > 0) {
            metrics.dma_bandwidth_mbps_used = (total_bytes_prefetched / total_time_sec) / (1024.0 * 1024.0);
        }
        
        // 计算CPU占用
        double total_cpu_cycles = metrics.total_ops * (config_.l1_latency_ns * config_.cpu_frequency_ghz / 1000.0);
        double available_cpu_cycles = config_.cpu_cores * config_.cpu_frequency_ghz * 1e9 * total_time_sec;
        if (available_cpu_cycles > 0) {
            metrics.cpu_usage_percent = (total_cpu_cycles / available_cpu_cycles) * 100.0;
            if (metrics.cpu_usage_percent > config_.cpu_cores * 100.0) {
                metrics.cpu_usage_percent = config_.cpu_cores * 100.0;
            }
        }
        
        return metrics;
    }
    
private:
    const B3Config& config_;
    size_t batch_size_;
    size_t cluster_size_bytes_;
    
    L1Cache l1_cache_;
    L2Cache l2_cache_;
    L3Cache l3_cache_;
    TokenDirectory directory_;
    
    std::queue<uint32_t> prefetch_queue_;
    std::mutex queue_mutex_;
    std::atomic<bool> stop_worker_;
    std::thread worker_thread_;
    std::atomic<uint64_t>* promotion_counter_ = nullptr;
    
    void schedule_prefetch(uint32_t token_id) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        prefetch_queue_.push(token_id);
    }
    
    void prefetch_worker() {
        while (!stop_worker_) {
            std::vector<uint32_t> batch;
            
            // 从队列中取出批量token
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                size_t count = std::min(prefetch_queue_.size(), batch_size_);
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
                if (cluster.empty()) continue;
                
                // 模拟DMA读取延迟
                size_t cluster_bytes = std::min(cluster_size_bytes_, cluster.size() * 16);
                uint64_t dma_latency_ns = config_.dma_setup_latency_ns + 
                    (cluster_bytes * 8 * 1e9) / (config_.dma_bandwidth_mbps * 1024.0 * 1024.0);
                simulate_delay_ns(dma_latency_ns);
                
                // 解析并安装到L1
                for (uint32_t token_id : cluster) {
                    Token token;
                    if (l2_cache_.peek(token_id, token)) {
                        if (l1_cache_.insert(token_id, token)) {
                            // 成功安装到L1
                            if (promotion_counter_) {
                                (*promotion_counter_)++;
                            }
                            // 更新目录
                            auto* dir_entry = directory_.get_entry(token_id);
                            if (dir_entry) {
                                dir_entry->layer = CacheLayer::L1;
                                dir_entry->version++;
                            }
                        }
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    void cluster_by_page(const std::vector<uint32_t>& batch,
                        std::vector<std::vector<uint32_t>>& clusters) {
        std::unordered_map<uint32_t, std::vector<uint32_t>> page_map;
        
        // 根据cluster_size_bytes_计算每页的token数量
        size_t tokens_per_page = cluster_size_bytes_ / 16;  // 假设每个token 16字节
        if (tokens_per_page == 0) tokens_per_page = 1;
        
        for (uint32_t token_id : batch) {
            uint32_t page_id = token_id / tokens_per_page;
            page_map[page_id].push_back(token_id);
        }
        
        for (auto& pair : page_map) {
            clusters.push_back(std::move(pair.second));
        }
    }
    
    void simulate_delay_ns(uint64_t delay_ns) {
        auto start = std::chrono::high_resolution_clock::now();
        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
            if (static_cast<uint64_t>(elapsed) >= delay_ns) break;
        }
    }
};

// 导出结果到JSON
void export_results(const std::string& filename, 
                   const std::vector<std::vector<B3Metrics>>& results,
                   const B3Config& config) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    
    out << "{\n";
    out << "  \"experiment\": \"B3 - Prefetched Batch: Batch Size and Cluster Strategy Impact\",\n";
    out << "  \"batch_sizes\": [8, 16, 32, 64, 128, 256],\n";
    out << "  \"cluster_sizes_kb\": [4, 16, 64, 256],\n";
    out << "  \"token_count\": " << config.token_count << ",\n";
    out << "  \"test_duration_sec\": " << config.test_duration_sec << ",\n";
    out << "  \"results\": [\n";
    
    bool first = true;
    for (size_t batch_idx = 0; batch_idx < 6; batch_idx++) {
        for (size_t cluster_idx = 0; cluster_idx < 4; cluster_idx++) {
            const auto& metrics = results[batch_idx][cluster_idx];
            
            if (!first) out << ",\n";
            first = false;
            
            out << "    {\n";
            out << "      \"batch_size\": " << config.batch_sizes[batch_idx] << ",\n";
            out << "      \"cluster_size_kb\": " << config.cluster_sizes_kb[cluster_idx] << ",\n";
            out << "      \"total_ops\": " << metrics.total_ops << ",\n";
            out << "      \"l1_hits\": " << metrics.l1_hits << ",\n";
            out << "      \"l2_hits\": " << metrics.l2_hits << ",\n";
            out << "      \"l3_hits\": " << metrics.l3_hits << ",\n";
            out << "      \"l1_hit_rate\": " << std::fixed << std::setprecision(4) 
                << metrics.l1_hit_rate << ",\n";
            out << "      \"l2_hit_rate\": " << std::fixed << std::setprecision(4) 
                << metrics.l2_hit_rate << ",\n";
            out << "      \"l3_hit_rate\": " << std::fixed << std::setprecision(4) 
                << metrics.l3_hit_rate << ",\n";
            out << "      \"l2_to_l1_promotions\": " << metrics.l2_to_l1_promotions << ",\n";
            out << "      \"prefetch_requests\": " << metrics.prefetch_requests << ",\n";
            out << "      \"l2_to_l1_success_rate\": " << std::fixed << std::setprecision(4) 
                << metrics.l2_to_l1_success_rate << ",\n";
            out << "      \"avg_latency_ns\": " << std::fixed << std::setprecision(2) 
                << metrics.avg_latency_ns << ",\n";
            out << "      \"throughput_ops_per_sec\": " << std::fixed << std::setprecision(2) 
                << metrics.throughput_ops_per_sec << ",\n";
            out << "      \"dma_bandwidth_mbps_used\": " << std::fixed << std::setprecision(2) 
                << metrics.dma_bandwidth_mbps_used << ",\n";
            out << "      \"cpu_usage_percent\": " << std::fixed << std::setprecision(2) 
                << metrics.cpu_usage_percent << "\n";
            out << "    }";
        }
    }
    
    out << "\n  ]\n";
    out << "}\n";
    out.close();
    std::cout << "Results exported to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    B3Config config;
    
    if (argc > 1) {
        config.output_file = argv[1];
    }
    
    // 确保输出目录存在
    size_t last_slash = config.output_file.find_last_of("/");
    if (last_slash != std::string::npos) {
        std::string dir = config.output_file.substr(0, last_slash);
        system(("mkdir -p " + dir).c_str());
    }
    
    std::cout << "========================================\n";
    std::cout << "Experiment B3: Prefetched Batch Test\n";
    std::cout << "========================================\n";
    std::cout << "Batch Sizes: 8, 16, 32, 64, 128, 256\n";
    std::cout << "Cluster Sizes: 4KB, 16KB, 64KB, 256KB\n";
    std::cout << "Token Count: " << config.token_count << "\n";
    std::cout << "Test Duration: " << config.test_duration_sec << " seconds per test\n";
    std::cout << "========================================\n\n";
    
    // 存储结果：batch_size[6] -> cluster_size[4]
    std::vector<std::vector<B3Metrics>> results(6);
    for (size_t i = 0; i < 6; i++) {
        results[i].resize(4);
    }
    
    // 运行所有参数组合测试
    for (size_t batch_idx = 0; batch_idx < 6; batch_idx++) {
        size_t batch_size = config.batch_sizes[batch_idx];
        
        for (size_t cluster_idx = 0; cluster_idx < 4; cluster_idx++) {
            size_t cluster_size_kb = config.cluster_sizes_kb[cluster_idx];
            
            std::cout << "Running test: batch_size=" << batch_size 
                      << ", cluster_size=" << cluster_size_kb << "KB ... ";
            std::cout.flush();
            
            PrefetchedBatchSimulator simulator(config, batch_size, cluster_size_kb);
            B3Metrics metrics = simulator.run_test();
            results[batch_idx][cluster_idx] = metrics;
            
            std::cout << "Done (L1 Hit Rate=" << std::fixed << std::setprecision(2)
                      << (metrics.l1_hit_rate * 100.0) << "%, "
                      << "L2->L1 Success=" << std::fixed << std::setprecision(2)
                      << (metrics.l2_to_l1_success_rate * 100.0) << "%, "
                      << "Avg Latency=" << metrics.avg_latency_ns << "ns, "
                      << "DMA BW=" << std::fixed << std::setprecision(2)
                      << metrics.dma_bandwidth_mbps_used << "MB/s)\n";
        }
    }
    
    // 导出结果
    export_results(config.output_file, results, config);
    
    std::cout << "\n========================================\n";
    std::cout << "Experiment B3 completed!\n";
    std::cout << "Results saved to: " << config.output_file << "\n";
    std::cout << "========================================\n";
    
    return 0;
}

