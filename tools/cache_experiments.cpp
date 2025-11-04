#include "simulator.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <getopt.h>

using namespace rdma_cache;

struct ExpArgs {
    std::string csv = "experiments.csv";
    size_t tokens = 10000;
    size_t accesses = 100000;
    int workload = 1;
    std::vector<size_t> l1_caps{2048, 4096, 8192};
    std::vector<int> batch_sizes{16, 32, 64, 128, 256};
    double pre_ratio = 0.0;
    int expand_window = 1;
};

static void parse_args(int argc, char** argv, ExpArgs& a) {
    static option long_opts[] = {
        {"csv", required_argument, 0, 'c'},
        {"tokens", required_argument, 0, 't'},
        {"accesses", required_argument, 0, 'a'},
        {"workload", required_argument, 0, 'w'},
        {"pre_ratio", required_argument, 0, 'r'},
        {"expand_window", required_argument, 0, 'e'},
        {0,0,0,0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "c:t:a:w:r:e:", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'c': a.csv = optarg; break;
            case 't': a.tokens = std::stoull(optarg); break;
            case 'a': a.accesses = std::stoull(optarg); break;
            case 'w': a.workload = std::stoi(optarg); break;
            case 'r': a.pre_ratio = std::stod(optarg); break;
            case 'e': a.expand_window = std::stoi(optarg); break;
            default: break;
        }
    }
}

static void write_header(std::ofstream& out) {
    out << "tokens,accesses,workload,l1_capacity,batch_size,pre_stage_ratio,expand_window,total,l1_hits,l2_hits,l3_hits,overall,l1_share,l2_share,l3_share,avg_latency_ns,total_estimated_bytes,estimated_comm_bw_MBps,num_expands,avg_expand_batch,promote_l3_to_l2,promote_l2_to_l1,demote_l1_to_l2,demote_l2_to_l3\n";
}

int main(int argc, char** argv) {
    ExpArgs args; parse_args(argc, argv, args);
    std::ofstream out(args.csv);
    write_header(out);

    for (size_t l1 : args.l1_caps) {
        for (int batch : args.batch_sizes) {
            ExperimentConfig cfg;
            cfg.num_tokens = args.tokens;
            cfg.num_accesses = args.accesses;
            cfg.workload_type = static_cast<WorkloadType>(args.workload);
            cfg.l1_capacity = l1;
            cfg.pre_stage_ratio = args.pre_ratio;
            cfg.expand_window = args.expand_window;

            Simulator sim(cfg);
            sim.run();
            auto m = sim.get_metrics();
            sim.export_results("results/exp_" + std::to_string(l1) + "_" + std::to_string(batch) + ".json");

            double total = static_cast<double>(m.total_accesses);
            double l1_share = total > 0 ? static_cast<double>(m.l1_hits) / total : 0.0;
            double l2_share = total > 0 ? static_cast<double>(m.l2_hits) / total : 0.0;
            double l3_share = total > 0 ? static_cast<double>(m.l3_hits) / total : 0.0;

            double avg_expand_batch = m.num_expands ? (double)m.total_expand_batch / m.num_expands : 0.0;

            out << args.tokens << "," << args.accesses << "," << args.workload << ","
                << l1 << "," << batch << "," << cfg.pre_stage_ratio << "," << cfg.expand_window << ","
                << m.total_accesses << "," << m.l1_hits << "," << m.l2_hits << "," << m.l3_hits << ","
                << m.overall_hit_rate() << "," << l1_share << "," << l2_share << "," << l3_share << ","
                << m.avg_latency_ns << "," << m.total_estimated_bytes << "," << m.estimated_comm_bw_MBps << ","
                << m.num_expands << "," << avg_expand_batch << ","
                << m.promote_l3_to_l2 << "," << m.promote_l2_to_l1 << ","
                << m.demote_l1_to_l2 << "," << m.demote_l2_to_l3
                << "\n";
        }
    }

    std::cout << "Wrote CSV: " << args.csv << std::endl;
    return 0;
}
