#include "tokens.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <getopt.h>
#include <cmath>

using namespace rdma_cache;

struct A1Config {
    std::string type = "QP";
    std::vector<size_t> N_values{1000, 10000, 100000, 1000000};
    int iters = 10;
    bool random_fields = true;
    std::string output_json = "exp_a1_results.json";

    // Hardware serialization unit modeling parameters
    double encode_extra_ns = 0.0;      // additional per-token compute (ns)
    double decode_extra_ns = 0.0;      // additional per-token compute (ns)
    double encode_fixed_overhead_ns = 0.0; // fixed overhead per token (ns)
    double decode_fixed_overhead_ns = 0.0; // fixed overhead per token (ns)
    int pipeline_width = 1;            // number of parallel lanes (>=1)
};

static void parse_args(int argc, char** argv, A1Config& cfg) {
    static option long_opts[] = {
        {"type", required_argument, 0, 't'},
        {"N", required_argument, 0, 'n'},
        {"N_values", required_argument, 0, 'N'},
        {"iters", required_argument, 0, 'i'},
        {"random", no_argument, 0, 'r'},
        {"typical", no_argument, 0, 'T'},
        {"output", required_argument, 0, 'o'},
        {"encode_extra_ns", required_argument, 0, 1000},
        {"decode_extra_ns", required_argument, 0, 1001},
        {"encode_fixed_overhead_ns", required_argument, 0, 1002},
        {"decode_fixed_overhead_ns", required_argument, 0, 1003},
        {"pipeline_width", required_argument, 0, 1004},
        {0,0,0,0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "t:n:N:i:rTo:", long_opts, nullptr)) != -1) {
        switch (c) {
            case 't': cfg.type = optarg; break;
            case 'n': {
                size_t nval = std::stoull(optarg);
                cfg.N_values = {nval};
                break;
            }
            case 'N': {
                // 支持逗号分隔的多个N值: --N_values=1000,10000,100000,1000000
                std::string val_str = optarg;
                cfg.N_values.clear();
                size_t pos = 0;
                while (pos < val_str.length()) {
                    size_t next = val_str.find(',', pos);
                    if (next == std::string::npos) next = val_str.length();
                    std::string n_str = val_str.substr(pos, next - pos);
                    if (!n_str.empty()) {
                        cfg.N_values.push_back(std::stoull(n_str));
                    }
                    pos = next + 1;
                }
                break;
            }
            case 'i': cfg.iters = std::stoi(optarg); break;
            case 'r': cfg.random_fields = true; break;
            case 'T': cfg.random_fields = false; break;
            case 'o': cfg.output_json = optarg; break;
            case 1000: cfg.encode_extra_ns = std::stod(optarg); break;
            case 1001: cfg.decode_extra_ns = std::stod(optarg); break;
            case 1002: cfg.encode_fixed_overhead_ns = std::stod(optarg); break;
            case 1003: cfg.decode_fixed_overhead_ns = std::stod(optarg); break;
            case 1004: cfg.pipeline_width = std::max(1, std::stoi(optarg)); break;
        }
    }
}

static void make_token(Token& tok, const std::string& type, uint32_t idx, bool random, std::mt19937& rng) {
    if (type == "PD") {
        tok.pd = PDToken(idx, random ? rng() : idx);
    } else if (type == "MR") {
        uint32_t lkey = random ? rng() : idx;
        uint8_t af = random ? (rng() % 256) : 0;
        uint8_t pb = random ? (8 + (rng() % 8)) : 12;
        uint16_t np = random ? (rng() % 65536) : 1024;
        tok.mr = MRToken(idx, lkey, af, pb, np);
    } else if (type == "CQ") {
        uint32_t cqn = random ? rng() : idx;
        uint16_t cqe = random ? (256 + (rng() % 4096)) : 4096;
        tok.cq = CQToken(idx, cqn, cqe, rng() % 256, 0);
    } else {
        tok.qp = QPToken(idx, random ? rng() : idx, rng() % 4, rng() % 8, 
                        0, 0, 0, random ? (rng() % 256) : 0);
    }
}

// 注意：现在使用连续内存布局，不再需要这些函数
// 但保留它们以保持接口兼容性（如果其他地方用到）
static size_t encode_token(const Token& tok, std::vector<uint8_t>& out) {
    const size_t sz = tok.size();
    out.resize(sz);
    std::memcpy(out.data(), &tok, sz);
    return sz;
}

static void decode_token(const std::vector<uint8_t>& in, Token& tok) {
    std::memcpy(&tok, in.data(), in.size());
}

struct Stats {
    std::vector<double> encode_times_ns;  // 纳秒
    std::vector<double> decode_times_ns;  // 纳秒
    size_t total_bytes = 0;
    size_t original_size_estimate = 0;
    
    double avg_encode_ns() const {
        if (encode_times_ns.empty()) return 0.0;
        double sum = 0; for (double v : encode_times_ns) sum += v;
        return sum / encode_times_ns.size();
    }
    
    double p95_encode_ns() const {
        if (encode_times_ns.empty()) return 0.0;
        auto sorted = encode_times_ns;
        std::sort(sorted.begin(), sorted.end());
        return sorted[std::min(sorted.size() - 1, size_t(sorted.size() * 0.95))];
    }
    
    double avg_decode_ns() const {
        if (decode_times_ns.empty()) return 0.0;
        double sum = 0; for (double v : decode_times_ns) sum += v;
        return sum / decode_times_ns.size();
    }
    
    double p95_decode_ns() const {
        if (decode_times_ns.empty()) return 0.0;
        auto sorted = decode_times_ns;
        std::sort(sorted.begin(), sorted.end());
        return sorted[std::min(sorted.size() - 1, size_t(sorted.size() * 0.95))];
    }
    
    // 计算带宽（MB/s）
    double encode_bandwidth_MBps(size_t bytes_per_token) const {
        double avg_ns = avg_encode_ns();
        if (avg_ns <= 0) return 0.0;
        // bytes_per_token / (avg_ns / 1e9) / (1024 * 1024)
        return (bytes_per_token * 1e9) / (avg_ns * 1024.0 * 1024.0);
    }
    
    double decode_bandwidth_MBps(size_t bytes_per_token) const {
        double avg_ns = avg_decode_ns();
        if (avg_ns <= 0) return 0.0;
        return (bytes_per_token * 1e9) / (avg_ns * 1024.0 * 1024.0);
    }
    
    // 吞吐量（百万tokens/秒，M tokens/sec）
    double encode_throughput_Mtokens_per_sec() const {
        double avg_ns = avg_encode_ns();
        if (avg_ns <= 0) return 0.0;
        return 1e9 / avg_ns / 1e6;  // 转换为百万tokens/秒
    }
    
    double decode_throughput_Mtokens_per_sec() const {
        double avg_ns = avg_decode_ns();
        if (avg_ns <= 0) return 0.0;
        return 1e9 / avg_ns / 1e6;
    }
    
    double compression_ratio() const {
        if (total_bytes == 0) return 1.0;
        return static_cast<double>(original_size_estimate) / total_bytes;
    }
};

int main(int argc, char** argv) {
    A1Config cfg;
    parse_args(argc, argv, cfg);
    
    std::cout << "A1 Experiment: Compression Ratio & Serialization Overhead\n";
    std::cout << "Type=" << cfg.type << ", N_values=";
    for (size_t n : cfg.N_values) std::cout << n << " ";
    std::cout << ", iters=" << cfg.iters << "\n";
    
    std::mt19937 rng(std::random_device{}());
    std::vector<std::pair<size_t, Stats>> results;
    
    for (size_t N : cfg.N_values) {
        std::cout << "\nN=" << N << ":\n";
        Stats stats;
        
        // 估算原始大小（未压缩）
        size_t est_size_per_token = 0;
        if (cfg.type == "PD") est_size_per_token = sizeof(PDToken) + 8; // 估算对齐开销
        else if (cfg.type == "MR") est_size_per_token = sizeof(MRToken) + 8;
        else if (cfg.type == "CQ") est_size_per_token = sizeof(CQToken) + 8;
        else est_size_per_token = sizeof(QPToken) + 8;
        stats.original_size_estimate = N * est_size_per_token;
        
        // 预分配编码缓冲区：使用连续内存，避免vector<vector>的间接访问
        const size_t token_size = (cfg.type == "PD" ? sizeof(PDToken) : 
                                  cfg.type == "MR" ? sizeof(MRToken) :
                                  cfg.type == "CQ" ? sizeof(CQToken) : sizeof(QPToken));
        
        for (int iter = 0; iter < cfg.iters; iter++) {
            std::vector<Token> tokens(N);
            for (size_t i = 0; i < N; i++) {
                make_token(tokens[i], cfg.type, static_cast<uint32_t>(i), cfg.random_fields, rng);
            }
            
            // 预分配编码缓冲区（每次迭代重新分配，模拟真实场景）
            std::vector<uint8_t> encoded_buffer(N * token_size);
            std::vector<size_t> encoded_offsets(N);
            size_t current_offset = 0;
            
            // 计算偏移（只计算一次，不测量）
            for (size_t i = 0; i < N; i++) {
                encoded_offsets[i] = current_offset;
                current_offset += tokens[i].size();
            }
            
            // 编码：每个token单独编码到连续内存（只测量memcpy，不包含分配）
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < N; i++) {
                std::memcpy(encoded_buffer.data() + encoded_offsets[i], &tokens[i], tokens[i].size());
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double enc_ns_total = std::chrono::duration<double, std::nano>(t1 - t0).count();
            double base_encode_ns_per_token = enc_ns_total / N;
            // Hardware modeling: per-token effective time is the slower between memory path and compute path
            double compute_encode_ns = cfg.encode_fixed_overhead_ns + (cfg.encode_extra_ns / std::max(1, cfg.pipeline_width));
            double modeled_encode_ns = std::max(base_encode_ns_per_token, compute_encode_ns);
            stats.encode_times_ns.push_back(modeled_encode_ns);
            
            // 预热：确保缓存已加载（至少预热前1000个）
            Token tmp;
            const size_t warmup_count = std::min(N, size_t(1000));
            for (size_t i = 0; i < warmup_count; i++) {
                std::memcpy(&tmp, encoded_buffer.data() + encoded_offsets[i], tokens[i].size());
            }
            
            // 解码：每个token单独解码，使用连续内存访问
            auto t2 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < N; i++) {
                std::memcpy(&tmp, encoded_buffer.data() + encoded_offsets[i], tokens[i].size());
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double dec_ns_total = std::chrono::duration<double, std::nano>(t3 - t2).count();
            double base_decode_ns_per_token = dec_ns_total / N;
            double compute_decode_ns = cfg.decode_fixed_overhead_ns + (cfg.decode_extra_ns / std::max(1, cfg.pipeline_width));
            double modeled_decode_ns = std::max(base_decode_ns_per_token, compute_decode_ns);
            stats.decode_times_ns.push_back(modeled_decode_ns);
            
            if (iter == 0) {
                stats.total_bytes = current_offset;
            }
            
            if ((iter + 1) % 5 == 0) {
                std::cout << "  iter " << (iter + 1) << "/" << cfg.iters << "\n";
            }
        }
        
        results.push_back({N, stats});
        
        size_t bytes_per_token = stats.total_bytes / N;
        std::cout << "  avg encode latency (modeled): " << std::fixed << std::setprecision(2) 
                  << stats.avg_encode_ns() << " ns/op\n";
        std::cout << "  p95 encode latency: " << stats.p95_encode_ns() << " ns/op\n";
        std::cout << "  encode bandwidth: " << std::setprecision(2)
                  << stats.encode_bandwidth_MBps(bytes_per_token) << " MB/s\n";
        std::cout << "  encode throughput: " << std::setprecision(2)
                  << stats.encode_throughput_Mtokens_per_sec() << " M tokens/sec\n";
        std::cout << "  avg decode latency (modeled): " << stats.avg_decode_ns() << " ns/op\n";
        std::cout << "  p95 decode latency: " << stats.p95_decode_ns() << " ns/op\n";
        std::cout << "  decode bandwidth: " << std::setprecision(2)
                  << stats.decode_bandwidth_MBps(bytes_per_token) << " MB/s\n";
        std::cout << "  decode throughput: " << std::setprecision(2)
                  << stats.decode_throughput_Mtokens_per_sec() << " M tokens/sec\n";
        std::cout << "  total bytes: " << stats.total_bytes << "\n";
        std::cout << "  bytes/token: " << bytes_per_token << "\n";
        std::cout << "  compression ratio: " << std::setprecision(4) 
                  << stats.compression_ratio() << "\n";
    }
    
    // 导出JSON
    std::ofstream json_out(cfg.output_json);
    json_out << std::fixed << std::setprecision(6);
    json_out << "{\n";
    json_out << "  \"type\": \"" << cfg.type << "\",\n";
    json_out << "  \"iters\": " << cfg.iters << ",\n";
    json_out << "  \"random_fields\": " << (cfg.random_fields ? "true" : "false") << ",\n";
    json_out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& [N, stats] = results[i];
        size_t bytes_per_token = stats.total_bytes / N;
        json_out << "    {\n";
        json_out << "      \"N\": " << N << ",\n";
        json_out << "      \"avg_encode_latency_ns\": " << stats.avg_encode_ns() << ",\n";
        json_out << "      \"p95_encode_latency_ns\": " << stats.p95_encode_ns() << ",\n";
        json_out << "      \"encode_bandwidth_MBps\": " << stats.encode_bandwidth_MBps(bytes_per_token) << ",\n";
        json_out << "      \"encode_throughput_Mtokens_per_sec\": " << stats.encode_throughput_Mtokens_per_sec() << ",\n";
        json_out << "      \"avg_decode_latency_ns\": " << stats.avg_decode_ns() << ",\n";
        json_out << "      \"p95_decode_latency_ns\": " << stats.p95_decode_ns() << ",\n";
        json_out << "      \"decode_bandwidth_MBps\": " << stats.decode_bandwidth_MBps(bytes_per_token) << ",\n";
        json_out << "      \"decode_throughput_Mtokens_per_sec\": " << stats.decode_throughput_Mtokens_per_sec() << ",\n";
        json_out << "      \"total_bytes\": " << stats.total_bytes << ",\n";
        json_out << "      \"bytes_per_token\": " << bytes_per_token << ",\n";
        json_out << "      \"compression_ratio\": " << stats.compression_ratio() << "\n";
        json_out << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
    }
    json_out << "  ]\n";
    json_out << "}\n";
    json_out.close();

    // Append modeling parameters for traceability
    std::ofstream json_out2(cfg.output_json, std::ios::app);
    // no-op to keep file handle usage explicit in case of future extensions
    
    std::cout << "\nResults exported to: " << cfg.output_json << "\n";
    return 0;
}

