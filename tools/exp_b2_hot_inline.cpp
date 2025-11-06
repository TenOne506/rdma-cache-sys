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

using namespace rdma_cache;

// 实验B2配置
struct B2Config {
    // 测试参数
    size_t concurrency_levels[5] = {1, 4, 16, 64, 256};  // 并发度
    size_t hot_token_count = 10;  // 热点token数量
    size_t ops_per_test = 1000000;  // 每次测试的操作次数
    size_t cas_retry_limit = 10;  // CAS最大重试次数
    
    // 延迟模拟（纳秒）
    uint64_t l1_access_latency_ns = 15;  // L1访问延迟（命中）
    uint64_t cas_latency_ns = 5;  // CAS操作延迟
    uint64_t cas_failure_penalty_ns = 20;  // CAS失败后的额外延迟
    
    // CAS失败率模拟（基于并发度）
    // 高并发时，CAS失败率会增加
    double get_cas_failure_rate(size_t concurrency) const {
        if (concurrency <= 1) return 0.01;  // 1%失败率
        if (concurrency <= 4) return 0.05;  // 5%失败率
        if (concurrency <= 16) return 0.15;  // 15%失败率
        if (concurrency <= 64) return 0.30;  // 30%失败率
        return 0.50;  // 50%失败率（高并发时）
    }
    
    // CPU配置（用于计算CPU占用率）
    uint32_t cpu_cores = 8;
    double cpu_frequency_ghz = 2.4;
    
    std::string output_file = "results/exp_b2_results.json";
};

// 性能指标
struct B2Metrics {
    std::vector<uint64_t> latencies;  // 所有延迟样本（纳秒）
    uint64_t total_ops = 0;
    uint64_t successful_inline_updates = 0;  // 成功inline更新次数
    uint64_t total_cas_retries = 0;  // CAS总重试次数
    uint64_t cas_failures = 0;  // CAS失败次数
    uint64_t fallback_count = 0;  // 降级到fallback的次数
    double throughput_ops_per_sec = 0.0;
    double cpu_usage_percent = 0.0;
    
    // 百分位数延迟
    double p50_latency_ns = 0.0;
    double p95_latency_ns = 0.0;
    double p99_latency_ns = 0.0;
    
    // CAS相关统计
    double avg_cas_retries_per_op = 0.0;
    double cas_failure_rate = 0.0;
    
    void calculate_percentiles() {
        if (latencies.empty()) return;
        
        std::sort(latencies.begin(), latencies.end());
        size_t n = latencies.size();
        
        p50_latency_ns = latencies[n * 50 / 100];
        p95_latency_ns = latencies[n * 95 / 100];
        p99_latency_ns = latencies[n * 99 / 100];
    }
    
    void reset() {
        latencies.clear();
        total_ops = 0;
        successful_inline_updates = 0;
        total_cas_retries = 0;
        cas_failures = 0;
        fallback_count = 0;
        throughput_ops_per_sec = 0.0;
        cpu_usage_percent = 0.0;
        p50_latency_ns = 0.0;
        p95_latency_ns = 0.0;
        p99_latency_ns = 0.0;
        avg_cas_retries_per_op = 0.0;
        cas_failure_rate = 0.0;
    }
};

// 模拟L1Entry的CAS操作
class MockL1Entry {
public:
    std::atomic<uint64_t> small_fields;
    std::atomic<uint32_t> version;
    std::atomic<uint32_t> freq;
    
    MockL1Entry() : small_fields(0), version(0), freq(0) {}
};

// Hot Inline模拟器
class HotInlineSimulator {
public:
    HotInlineSimulator(const B2Config& config) : config_(config) {
        // 初始化热点tokens
        for (size_t i = 0; i < config_.hot_token_count; i++) {
            hot_tokens_[i] = std::make_unique<MockL1Entry>();
        }
    }
    
    B2Metrics run_test(size_t concurrency) {
        B2Metrics metrics;
        metrics.latencies.reserve(config_.ops_per_test);
        
        std::atomic<uint64_t> completed_ops(0);
        std::atomic<uint64_t> successful_updates(0);
        std::atomic<uint64_t> total_retries(0);
        std::atomic<uint64_t> cas_failures(0);
        std::atomic<uint64_t> fallbacks(0);
        std::vector<std::thread> threads;
        std::mutex metrics_mutex;
        
        // 获取CAS失败率
        double cas_failure_rate = config_.get_cas_failure_rate(concurrency);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 启动工作线程
        for (size_t t = 0; t < concurrency; t++) {
            threads.emplace_back([&, t, cas_failure_rate]() {
                std::mt19937 rng(std::random_device{}() + t);
                std::uniform_real_distribution<double> failure_dist(0.0, 1.0);
                std::uniform_int_distribution<size_t> token_dist(0, config_.hot_token_count - 1);
                
                size_t ops_per_thread = config_.ops_per_test / concurrency;
                if (t == 0) {
                    ops_per_thread += config_.ops_per_test % concurrency;
                }
                
                for (size_t i = 0; i < ops_per_thread; i++) {
                    // 随机选择一个热点token
                    uint32_t token_id = static_cast<uint32_t>(token_dist(rng));
                    auto& entry = hot_tokens_[token_id];
                    
                    auto op_start = std::chrono::high_resolution_clock::now();
                    
                    // 模拟L1访问延迟
                    simulate_delay_ns(config_.l1_access_latency_ns);
                    
                    // 尝试CAS更新
                    uint64_t latency_ns = config_.l1_access_latency_ns;
                    uint64_t cas_retries = 0;
                    bool cas_success = false;
                    
                    // 模拟CAS操作（可能失败）
                    for (uint64_t retry = 0; retry < config_.cas_retry_limit; retry++) {
                        simulate_delay_ns(config_.cas_latency_ns);
                        cas_retries++;
                        
                        // 根据失败率决定是否成功
                        bool will_fail = (failure_dist(rng) < cas_failure_rate);
                        
                        if (!will_fail) {
                            // CAS成功
                            uint64_t old_val = entry->small_fields.load();
                            uint64_t new_val = old_val + 1;  // 简单的增量操作
                            if (entry->small_fields.compare_exchange_weak(old_val, new_val)) {
                                cas_success = true;
                                entry->freq++;
                                break;
                            }
                            // CAS失败（值已改变），继续重试
                            latency_ns += config_.cas_failure_penalty_ns;
                            cas_failures++;
                        } else {
                            // CAS失败（由于竞争）
                            latency_ns += config_.cas_failure_penalty_ns;
                            cas_failures++;
                            
                            // 如果达到重试上限，降级到fallback
                            if (retry == config_.cas_retry_limit - 1) {
                                fallbacks++;
                                latency_ns += config_.l1_access_latency_ns * 2;  // fallback额外延迟
                                break;
                            }
                        }
                    }
                    
                    auto op_end = std::chrono::high_resolution_clock::now();
                    uint64_t total_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        op_end - op_start).count();
                    
                    // 使用计算的延迟而不是实际测量的延迟（因为忙等待模拟）
                    total_latency_ns = latency_ns;
                    
                    {
                        std::lock_guard<std::mutex> lock(metrics_mutex);
                        metrics.latencies.push_back(total_latency_ns);
                    }
                    
                    if (cas_success) {
                        successful_updates++;
                    }
                    total_retries += cas_retries;
                    completed_ops++;
                }
            });
        }
        
        // 等待所有线程完成
        for (auto& t : threads) {
            t.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        metrics.total_ops = completed_ops;
        metrics.successful_inline_updates = successful_updates;
        metrics.total_cas_retries = total_retries;
        metrics.cas_failures = cas_failures;
        metrics.fallback_count = fallbacks;
        metrics.calculate_percentiles();
        
        // 计算吞吐量
        double total_time_sec = duration / 1000.0;
        if (total_time_sec > 0) {
            metrics.throughput_ops_per_sec = metrics.total_ops / total_time_sec;
        }
        
        // 计算CAS相关统计
        if (metrics.total_ops > 0) {
            metrics.avg_cas_retries_per_op = static_cast<double>(metrics.total_cas_retries) / metrics.total_ops;
            metrics.cas_failure_rate = static_cast<double>(metrics.cas_failures) / (metrics.cas_failures + metrics.successful_inline_updates);
        }
        
        // 计算CPU占用
        double total_cpu_cycles = metrics.total_ops * (config_.l1_access_latency_ns * config_.cpu_frequency_ghz / 1000.0);
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
    const B2Config& config_;
    std::unordered_map<uint32_t, std::unique_ptr<MockL1Entry>> hot_tokens_;
    
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
                   const std::vector<B2Metrics>& results,
                   const B2Config& config) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    
    out << "{\n";
    out << "  \"experiment\": \"B2 - Hot Inline Path: Low Latency & Concurrent Contention Test\",\n";
    out << "  \"hot_token_count\": " << config.hot_token_count << ",\n";
    out << "  \"ops_per_test\": " << config.ops_per_test << ",\n";
    out << "  \"concurrency_levels\": [1, 4, 16, 64, 256],\n";
    out << "  \"results\": [\n";
    
    for (size_t i = 0; i < results.size(); i++) {
        const auto& metrics = results[i];
        
        if (i > 0) out << ",\n";
        
        out << "    {\n";
        out << "      \"concurrency\": " << config.concurrency_levels[i] << ",\n";
        out << "      \"p50_latency_ns\": " << metrics.p50_latency_ns << ",\n";
        out << "      \"p95_latency_ns\": " << metrics.p95_latency_ns << ",\n";
        out << "      \"p99_latency_ns\": " << metrics.p99_latency_ns << ",\n";
        out << "      \"throughput_ops_per_sec\": " << std::fixed << std::setprecision(2) 
            << metrics.throughput_ops_per_sec << ",\n";
        out << "      \"successful_inline_updates\": " << metrics.successful_inline_updates << ",\n";
        out << "      \"successful_inline_updates_per_sec\": " << std::fixed << std::setprecision(2)
            << (metrics.successful_inline_updates > 0 ? metrics.throughput_ops_per_sec * 
                (static_cast<double>(metrics.successful_inline_updates) / metrics.total_ops) : 0.0) << ",\n";
        out << "      \"total_cas_retries\": " << metrics.total_cas_retries << ",\n";
        out << "      \"cas_retries_per_sec\": " << std::fixed << std::setprecision(2)
            << (metrics.throughput_ops_per_sec * metrics.avg_cas_retries_per_op) << ",\n";
        out << "      \"avg_cas_retries_per_op\": " << std::fixed << std::setprecision(3)
            << metrics.avg_cas_retries_per_op << ",\n";
        out << "      \"cas_failures\": " << metrics.cas_failures << ",\n";
        out << "      \"cas_failure_rate\": " << std::fixed << std::setprecision(4)
            << metrics.cas_failure_rate << ",\n";
        out << "      \"fallback_count\": " << metrics.fallback_count << ",\n";
        out << "      \"cpu_usage_percent\": " << std::fixed << std::setprecision(2)
            << metrics.cpu_usage_percent << ",\n";
        out << "      \"total_ops\": " << metrics.total_ops << "\n";
        out << "    }";
    }
    
    out << "\n  ]\n";
    out << "}\n";
    out.close();
    std::cout << "Results exported to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    B2Config config;
    
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
    std::cout << "Experiment B2: Hot Inline Path Test\n";
    std::cout << "========================================\n";
    std::cout << "Hot Token Count: " << config.hot_token_count << "\n";
    std::cout << "Operations per test: " << config.ops_per_test << "\n";
    std::cout << "Concurrency Levels: 1, 4, 16, 64, 256\n";
    std::cout << "========================================\n\n";
    
    // 存储结果
    std::vector<B2Metrics> results;
    HotInlineSimulator simulator(config);
    
    // 运行所有并发度测试
    for (size_t concurrency : config.concurrency_levels) {
        std::cout << "Running test with concurrency=" << concurrency << " ... ";
        std::cout.flush();
        
        B2Metrics metrics = simulator.run_test(concurrency);
        results.push_back(metrics);
        
        std::cout << "Done (P50=" << metrics.p50_latency_ns << "ns, "
                  << "P95=" << metrics.p95_latency_ns << "ns, "
                  << "P99=" << metrics.p99_latency_ns << "ns, "
                  << "Throughput=" << std::fixed << std::setprecision(2)
                  << metrics.throughput_ops_per_sec << "ops/s, "
                  << "CAS Retries=" << std::fixed << std::setprecision(2)
                  << metrics.avg_cas_retries_per_op << "/op, "
                  << "CAS Failure Rate=" << std::fixed << std::setprecision(2)
                  << (metrics.cas_failure_rate * 100.0) << "%)\n";
    }
    
    // 导出结果
    export_results(config.output_file, results, config);
    
    std::cout << "\n========================================\n";
    std::cout << "Experiment B2 completed!\n";
    std::cout << "Results saved to: " << config.output_file << "\n";
    std::cout << "========================================\n";
    
    return 0;
}

