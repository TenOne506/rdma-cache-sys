#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>

namespace rdma_cache {

// Token类型枚举
enum class TokenType : uint8_t {
    PD = 0,
    MR = 1,
    CQ = 2,
    QP = 3
};

// 统一化Token头部结构
#pragma pack(push, 1)
struct RDMAObjTokenHdr {
    uint8_t type;        // 0=PD,1=MR,2=CQ,3=QP
    uint8_t version;     // 结构版本号
    uint16_t index;      // 在池中的索引
    
    RDMAObjTokenHdr() : type(0), version(1), index(0) {}
    RDMAObjTokenHdr(TokenType t, uint16_t idx) 
        : type(static_cast<uint8_t>(t)), version(1), index(idx) {}
};

// PD Token
struct PDToken {
    RDMAObjTokenHdr hdr;
    uint32_t pdn;        // 硬件PD号
    
    PDToken() : hdr(), pdn(0) {}
    PDToken(uint16_t idx, uint32_t pdn_val) 
        : hdr(TokenType::PD, idx), pdn(pdn_val) {}
    
    size_t size() const { return sizeof(PDToken); }
    std::string to_string() const;
};

// MR Token
struct MRToken {
    RDMAObjTokenHdr hdr;
    uint32_t lkey;        // 本地 key
    uint8_t  access_flags;
    uint8_t  page_bits;   // page_shift
    uint16_t n_pages;     // 页数量
    
    MRToken() : hdr(), lkey(0), access_flags(0), page_bits(12), n_pages(0) {}
    MRToken(uint16_t idx, uint32_t lk, uint8_t af, uint8_t pb, uint16_t np)
        : hdr(TokenType::MR, idx), lkey(lk), access_flags(af), 
          page_bits(pb), n_pages(np) {}
    
    size_t size() const { return sizeof(MRToken); }
    std::string to_string() const;
};

// CQ Token
struct CQToken {
    RDMAObjTokenHdr hdr;
    uint32_t cqn;        // 硬件CQ号
    uint16_t cqe;        // CQ entries数
    uint8_t  comp_vec;   // completion vector索引
    uint8_t  flags;      // 压缩标志或额外状态
    
    CQToken() : hdr(), cqn(0), cqe(0), comp_vec(0), flags(0) {}
    CQToken(uint16_t idx, uint32_t cqn_val, uint16_t cqe_val, 
            uint8_t cv, uint8_t fl)
        : hdr(TokenType::CQ, idx), cqn(cqn_val), cqe(cqe_val), 
          comp_vec(cv), flags(fl) {}
    
    size_t size() const { return sizeof(CQToken); }
    std::string to_string() const;
};

// QP Token
struct QPToken {
    RDMAObjTokenHdr hdr;
    uint32_t qpn;          // 硬件QP号
    uint8_t  qp_type;      // IB_QPT_RC, UD等
    uint8_t  state;        // RTS, SQD等
    uint8_t  pd_index;     // PD索引
    uint8_t  scq_index;    // Send CQ索引
    uint8_t  rcq_index;    // Recv CQ索引
    uint8_t  access_flags; // QP权限
    
    QPToken() : hdr(), qpn(0), qp_type(0), state(0), 
                pd_index(0), scq_index(0), rcq_index(0), access_flags(0) {}
    QPToken(uint16_t idx, uint32_t qpn_val, uint8_t qt, uint8_t st,
            uint8_t pdi, uint8_t sci, uint8_t rci, uint8_t af)
        : hdr(TokenType::QP, idx), qpn(qpn_val), qp_type(qt), state(st),
          pd_index(pdi), scq_index(sci), rcq_index(rci), access_flags(af) {}
    
    size_t size() const { return sizeof(QPToken); }
    std::string to_string() const;
};

#pragma pack(pop)

// Token基类（使用union存储不同类型）
union Token {
    RDMAObjTokenHdr hdr;
    PDToken pd;
    MRToken mr;
    CQToken cq;
    QPToken qp;
    
    Token() : hdr() {}
    
    size_t size() const {
        switch (static_cast<TokenType>(hdr.type)) {
            case TokenType::PD: return pd.size();
            case TokenType::MR: return mr.size();
            case TokenType::CQ: return cq.size();
            case TokenType::QP: return qp.size();
            default: return sizeof(RDMAObjTokenHdr);
        }
    }
    
    std::string to_string() const {
        switch (static_cast<TokenType>(hdr.type)) {
            case TokenType::PD: return pd.to_string();
            case TokenType::MR: return mr.to_string();
            case TokenType::CQ: return cq.to_string();
            case TokenType::QP: return qp.to_string();
            default: return "UnknownToken";
        }
    }
};

} // namespace rdma_cache

