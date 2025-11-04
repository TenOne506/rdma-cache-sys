#include "tokens.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <getopt.h>

using namespace rdma_cache;

struct Args {
    std::string type = "QP";
    size_t N = 100000;
    int iters = 10;
    int threads = 1;
};

static void parse_args(int argc, char** argv, Args& a) {
    static option long_opts[] = {
        {"type", required_argument, 0, 't'},
        {"N", required_argument, 0, 'n'},
        {"iters", required_argument, 0, 'i'},
        {"threads", required_argument, 0, 'p'},
        {0,0,0,0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "t:n:i:p:", long_opts, nullptr)) != -1) {
        switch (c) {
            case 't': a.type = optarg; break;
            case 'n': a.N = std::stoull(optarg); break;
            case 'i': a.iters = std::stoi(optarg); break;
            case 'p': a.threads = std::stoi(optarg); break;
            default: break;
        }
    }
}

static void make_token(Token& tok, const std::string& type, uint32_t idx) {
    if (type == "PD") {
        tok.pd = PDToken(idx, idx);
    } else if (type == "MR") {
        tok.mr = MRToken(idx, idx, 0, 12, 1024);
    } else if (type == "CQ") {
        tok.cq = CQToken(idx, idx, 4096, 0, 0);
    } else {
        tok.qp = QPToken(idx, idx, 0, 0, 0, 0, 0, 0);
    }
}

static size_t encode_token(const Token& tok, std::vector<uint8_t>& out) {
    const size_t sz = tok.size();
    out.resize(sz);
    std::memcpy(out.data(), &tok, sz);
    return sz;
}

static void decode_token(const std::vector<uint8_t>& in, Token& tok) {
    std::memcpy(&tok, in.data(), in.size());
}

int main(int argc, char** argv) {
    Args args; parse_args(argc, argv, args);

    std::cout << "bench_token_compress: type=" << args.type
              << " N=" << args.N << " iters=" << args.iters
              << " threads=" << args.threads << std::endl;

    std::vector<Token> tokens(args.N);
    for (size_t i = 0; i < args.N; ++i) {
        make_token(tokens[i], args.type, static_cast<uint32_t>(i));
    }

    std::atomic<size_t> total_bytes{0};
    std::atomic<size_t> completed{0};

    auto worker = [&](size_t start, size_t end) {
        std::vector<uint8_t> buf;
        Token tmp;
        for (int it = 0; it < args.iters; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            size_t local_bytes = 0;
            for (size_t i = start; i < end; ++i) {
                local_bytes += encode_token(tokens[i], buf);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            for (size_t i = start; i < end; ++i) {
                decode_token(buf, tmp);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            total_bytes += local_bytes;
            completed++;

            double enc_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double dec_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
            // 打印每次迭代线程本地测量（可选）
            // std::cout << "thread segment enc_us=" << enc_us << " dec_us=" << dec_us << std::endl;
        }
    };

    std::vector<std::thread> threads;
    size_t per = (args.N + args.threads - 1) / args.threads;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < args.threads; ++p) {
        size_t s = p * per;
        size_t e = std::min(args.N, s + per);
        if (s >= e) break;
        threads.emplace_back(worker, s, e);
    }
    for (auto& th : threads) th.join();
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    double ops = static_cast<double>(args.N) * args.iters;

    std::cout << "total_bytes=" << total_bytes.load() << std::endl;
    std::cout << "avg_us_per_encode_decode=" << (total_us / ops) << std::endl;
    std::cout << "throughput_tokens_per_sec=" << (ops * 1e6 / total_us) << std::endl;

    return 0;
}
