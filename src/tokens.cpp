#include "tokens.hpp"
#include <sstream>

namespace rdma_cache {

std::string PDToken::to_string() const {
    std::ostringstream oss;
    oss << "PDToken(idx=" << hdr.index << ", pdn=" << pdn << ")";
    return oss.str();
}

std::string MRToken::to_string() const {
    std::ostringstream oss;
    oss << "MRToken(idx=" << hdr.index << ", lkey=" << lkey 
        << ", pages=" << n_pages << ")";
    return oss.str();
}

std::string CQToken::to_string() const {
    std::ostringstream oss;
    oss << "CQToken(idx=" << hdr.index << ", cqn=" << cqn 
        << ", cqe=" << cqe << ")";
    return oss.str();
}

std::string QPToken::to_string() const {
    std::ostringstream oss;
    oss << "QPToken(idx=" << hdr.index << ", qpn=" << qpn 
        << ", type=" << (int)qp_type << ", state=" << (int)state << ")";
    return oss.str();
}

} // namespace rdma_cache

