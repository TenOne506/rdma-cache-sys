## RDMA 缓存优化及仿真系统


#### 元数据的压缩

##### 统一化token结构

```C++
#pragma pack(push, 1)
struct RDMAObjTokenHdr {
    uint8_t type;        // 0=PD,1=MR,2=CQ,3=QP
    uint8_t version;     // 结构版本号
    uint16_t index;      // 在池中的索引
};
```

#### PDToken

```C++
struct PDToken {
    RDMAObjTokenHdr hdr;
    uint32_t pdn;        // 硬件PD号
};
```

##### MRToken

```C++
struct MRToken {
    RDMAObjTokenHdr hdr;
    uint32_t lkey;        // 本地 key
    uint8_t  access_flags;
    uint8_t  page_bits;   // page_shift
    uint16_t n_pages;     // 页数量
};
```

##### CQToken

```C++
struct CQToken {
    RDMAObjTokenHdr hdr;
    uint32_t cqn;        // 硬件CQ号
    uint16_t cqe;        // CQ entries数
    uint8_t  comp_vec;   // completion vector索引
    uint8_t  flags;      // 压缩标志或额外状态
};
```

##### QPToken

```C++
struct QPToken {
    RDMAObjTokenHdr hdr;
    uint32_t qpn;          // 硬件QP号
    uint8_t  qp_type;      // IB_QPT_RC, UD等
    uint8_t  state;        // RTS, SQD等
    uint8_t  pd_index;     // PD索引
    uint8_t  scq_index;    // Send CQ索引
    uint8_t  rcq_index;    // Recv CQ索引
    uint8_t  access_flags; // QP权限
};
```

#### 分层缓存设计

- L1 — NIC 本地 SRAM（极热）

  - 存放最频繁访问的小结构：doorbell 缓冲、active QP 的活跃字段、热点 WQE cache、短期状态机变量。

  - 要求低纳秒级延迟；容量受限（几十 KB ~ 几 MB 级，设备相关）。

- L2 — CXL 设备内存（热 & 可池化）

  - 存放大但仍有较高访问率的表格：MR descriptor table、QP descriptor 的大字段、hash bucket、索引/目录结构、批量 WQE 队列。
  - 容量可扩展（GB 级），延迟介于 SRAM 与主 DRAM 之间，但比主机跨 PCIe/DMA 的往返更优/或在可接受范围内（取决于实现）。
  - 支撑设备直接 load/store（若硬件支持）或低开销的 DMA。

- L3 — Host DRAM（冷）
  - 存放不常访问的历史记录、管理信息、持久化元信息、或作为回退/同步存储。




####  数据放置与热/冷分层策略

###### 初始放置规则

- MR / QP 创建时，默认把描述符分配在 L3（host DRAM），并在驱动层维护一份轻量索引（例如 32-bit compact handle/ID）。
- 当 MR / QP 被频繁访问（阈值由计数器或采样决定），将条目迁移/promote 到 L2（CXL）；对极热条目再 promote 到 L1（本地 SRAM）用于超低延迟访问。

###### 热度判定与迁移策略

- 使用基于计数器的热度（例如 sliding-window hit count）或轻量采样（每 N 次访问采样一次）来决定是否 promote/demote。
- 引入 **LRU / LFU 混合** 的替换策略：L1 使用 LFU 优先保持长时间热点，L2 使用 LRU 以保证大容量条目有机会被替换。
- 迁移时：采用批量迁移（把相邻或同 bucket 的条目一起迁移）以 amortize fetch 成本，并在迁移时预取邻近若干条目（spatial prefetch）。

###### 索引压缩与句柄设计

- 使用紧凑句柄（32-bit）作为外部引用，内部映射到 L2/L3 的物理地址或偏移。句柄表放在 host driver 及 NIC 的轻量索引中（可复制到 L2 的小目录）。
- 在 L2 使用 delta 编码或 compact bucket 表（例如 open-addressing / compact chained buckets）以减少内存占用并提高缓存局部性。


#### 访问模式优化

##### 访问模式伪代码框架与实现

#####  访问模式伪代码框架与实现要点（Hot Inline / Prefetched Batch / On-demand Expand）

下面给出三种访问模式的 **数据结构摘要**、每种模式的**高阶伪代码**（便于直接移植到固件/驱动实现），以及实现时的关键注意点（并发、一致性、性能调优）。

---

###### 共享数据结构（全局、需 原子/并发 安全）

```c
// 句柄目录项（位于 L2 或 driver 的小目录页；非常小，便于缓存）
struct TokenDirectoryEntry {
    uint32_t token_offset;   // L2 或 L3 中的偏移 / 物理地址
    uint8_t  layer;          // 1=L1,2=L2,3=L3 (表示当前主副本层)
    uint8_t  flags;          // bit flags: pinned, dirty, migrating, etc.
    uint16_t reserved;
    // optional: version / seq number for consistency
    uint32_t version;
};

// L1 缓存条目（位于 NIC SRAM，本地）
struct L1Entry {
    uint32_t token_id;       // compact id
    uint64_t small_fields;   // packed hot fields (fits inline)
    uint32_t metadata;       // e.g., version / flags
    // eviction metadata:
    uint32_t freq;           // approximate frequency counter
    uint64_t last_access_ts; // for LRU/LFU hybrid
};

// L2 描述页（位于 CXL.mem，存放压缩 token 数据，按 page/bucket 聚簇）
struct CompressedPageHdr {
    uint32_t page_id;
    uint16_t used_slots;
    uint16_t slot_bitmap;    // or larger bitmap
    // followed by compact token byte streams
};
```

> 实现要点：`TokenDirectory` 必须能原子地从 driver 与 NIC 间共享（通过映射的页或小型共享命令）。`L1Entry` 的更新必须使用原子写或 lock-free CAS + versioning 保证并发正确。

---

###### A. Hot Inline 模式（L1 快速路径）

**目标：** 尽可能在本地 L1 用单次查找/原子更新完成控制操作，避免任何 L2/L3 访问。

 伪代码（NIC 固件/FPGA 侧）

```c
// Called when processing a WR that references token_id
function process_wr_hot_inline(token_id, operation, op_args):
    entry = L1_lookup(token_id)
    if likely(entry != NULL):
        // perform inline checks / updates atomically
        // use small atomic compare-and-swap or CAS-based update
        old_version = entry.version
        new_small_fields = apply_op_inline(entry.small_fields, operation, op_args)
        if CAS(&entry.small_fields, entry.small_fields, new_small_fields):
            // update hist/stat
            entry.freq += 1
            entry.last_access_ts = now()
            return SUCCESS
        else:
            // contention: fallback to lock or retry limited times
            retry_inline_or_fallback(token_id, operation, op_args)
    else:
        // L1 miss: fall through to higher-level handler
        return HOT_INLINE_MISS
```

注意点

* `apply_op_inline` 只修改那些被设计为“可原子更新的小字段”。复杂字段不能在此路径更改。
* 对于高冲突的字段，采用 CAS+version 或 small lock；若 CAS 失败次数超过阈值，降级为 `Prefetched Batch` 或 `On-demand Expand` 路径以避免长时间 spinning。
* 统计（`freq` / `last_access_ts`）用于后续 promote/demote 决策；对统计更新可使用近似计数（如小位增长器）以减少开销。

---

###### B. Prefetched Batch 模式（L2 批量预取路径）

**目标：** 把多个 token 的访问合并成一批 DMA 请求，从 L2 拉回到 L1 或临时缓冲区; 适用于多 token 顺序或短时间内重复访问场景。

伪代码（NIC 固件 + Prefetch Scheduler）

```c
// Scheduler accumulates misses and signals batch fetches
global prefetch_queue  // lock-free queue of token_ids

function schedule_prefetch(token_id):
    enqueue(prefetch_queue, token_id)
    if queue_size(prefetch_queue) >= PREFETCH_BATCH_THRESHOLD:
        trigger_prefetch_worker()

function prefetch_worker():
    batch = drain(prefetch_queue, MAX_BATCH)
    // cluster batch by compressed page / contiguous offsets to maximize DMA efficiency
    clusters = cluster_by_page(batch)
    for each cluster in clusters:
        // issue single DMA read for cluster.range -> temp_buf
        dma_read(cluster.page_addr, temp_buf, cluster.size)
        // parse temp_buf into token entries
        for each token in cluster:
            token_data = decode_compressed_token(temp_buf, token.offset)
            // install into L1 (or L1 staging area)
            install_to_L1_or_staging(token.id, token_data)
            update_directory(token.id, layer=L1, version=...)
    // wake up pending WR processors (if they waited)
```

 注意点

* **聚类（cluster_by_page）**：把分散的 token_id 按物理页分组，做到以页为单位的 DMA 聚合，显著提高带宽利用率。
* **批量大小**：`PREFETCH_BATCH_THRESHOLD` 和 `MAX_BATCH` 需要硬件/平台测定；典型初始值：32–256 token/批。
* **回压（backpressure）**：当 DMA 带宽占满时，prefetch 队列要提供 backpressure 通知上游（或丢弃次优预fetch）。
* **安装策略**：prefetch 的 token 可直接写入 L1，或写入 L1 的 staging 区以避免冲掉更热的条目；安装时执行替换策略（LFU/LRU 混合）。

---

###### C. On-demand Expand 模式（L3 回退路径：解压 + 插入）

**目标：** 当 token 在 L1/L2 都缺失，或需要完整展开字段时，从 L3（host DRAM）或长期存储按需 fetch 并解压，然后更新 L2/L1。

伪代码（NIC 固件 + Driver 协作）

```c
function handle_cache_miss_expand(token_id, wait_for_completion):
    // 1. Request directory info (if not present or marked L3)
    dir_entry = read_shared_directory(token_id)
    if dir_entry.layer == L3:
        // create an expand request to host driver (via doorbell / mailbox)
        req = create_expand_request(token_id, dir_entry.token_offset, requester_id)
        send_expand_request_to_host(req)
        if wait_for_completion:
            wait_for(req.completion)  // blocking or async completion callback
            // host will DMA/serialize compressed page into CXL or directly to NIC buffer
            token_data = req.result  // full expanded token bytes
            // optional: partially decompress on NIC (if supported) or store compressed page in L2
            install_token_after_expand(token_id, token_data)
            return SUCCESS
        else:
            return EXPAND_ISSUED  // non-blocking path; WR will be retried later
    else:
        // L2 says present but L1 missed: schedule prefetch
        schedule_prefetch(token_id)
        return PREFETCH_ISSUED
```

 Driver（Host）侧伪代码

```c
// Host receives expand request (from NIC)
function host_handle_expand(req):
    // read compressed page from host DRAM
    compressed_page = read_host_page(req.page_addr)
    // optionally perform decompression to a compact form suitable for NIC,
    // or stage compressed_page in CXL.mem for NIC DMA to pick up
    if NIC_supports_direct_dma_to_CXL:
        write_to_CXL_page(req.page_addr, compressed_page)
    else:
        perform DMA to NIC local buffer and notify
    signal_completion(req)
```

注意点

* Expand 通常需要 driver 协作（doorbell / mailbox / admin queue）。为了性能，driver 可以选择把热点页 pre-stage 到 CXL，减少后续 expand 延迟。
* Expand 请求应支持优先级：紧急（阻塞应用）与 背景（异步填充）两种。
* 安全性：expand 涉及主机内存读写，必须走 IOMMU / 权限校验流程，避免越权。

---

##### 3.3.2 — Promote / Demote 状态迁移规则（策略、阈值、伪代码）

将 token 在 L1/L2/L3 之间迁移的逻辑是系统性能的核心。下列给出一个可工程实现的、参数化的 promote/demote 策略，适合写进论文的“算法/实现”段落，并给出伪代码、参数默认值与性能影响说明。

---

##### 基本概念与元数据（每 token）

* `freq_counter`：近似访问频率（滑动窗口计数、或分段计数器）。
* `last_access_ts`：最近访问时间戳（用于 LRU 成分）。
* `state` ∈ {L1, L2, L3, MIGRATING_TO_L1, MIGRATING_TO_L2, MIGRATING_TO_L3}。
* `pinned`：布尔，表示是否被固定（例如 admin/pinned）。
* `dirty`：若在 L1 修改后未回写至 L2，置脏位。

---

##### 参数建议（初始默认值，需在实验中调优）

* `PROMOTE_TO_L1_THRESHOLD = 128` // freq hits within window
* `PROMOTE_TO_L2_THRESHOLD = 16`  // freq hits within window
* `DEMOTE_FROM_L1_IDLE_NS = 1_000_000` // 1ms 未访问则考虑 demote
* `DEMOTE_FROM_L2_IDLE_NS = 10_000_000` // 10ms
* `L1_CAPACITY = platform_dependent` // e.g., 8K entries
* `L2_PAGE_PREFETCH_SIZE = 64 * 1024` // 64KB clustering

> 这些值依硬件与 workload 强烈相关，应在论文中作为参数扫描项呈现。

---

##### Promote / Demote 高阶规则（策略说明）

1. **Promote to L2**：当 `freq_counter >= PROMOTE_TO_L2_THRESHOLD` 且当前处于 L3（或 large cold region），将 token 的压缩页或 token 本身迁移到 L2（CXL），并将目录 `layer` 更新为 L2。迁移以页/cluster 为单位。
2. **Promote to L1**：当 `freq_counter >= PROMOTE_TO_L1_THRESHOLD` 且 token 在 L2 或 L3，发起 L2→L1 的迁移；L1 插入使用替换策略（见下）。若 L1 空间不足，触发 L1 Eviction。
3. **Demote from L1**：若 token `last_access_ts` 超过 `DEMOTE_FROM_L1_IDLE_NS` 或 L1 需要空间用于更热条目，且 token 未被 `pinned`，将 token 写回/merge 到 L2（若 `dirty` 则写回差分），更新目录为 L2。
4. **Demote from L2**：若 token 在 L2 且长时间不访问（超过 `DEMOTE_FROM_L2_IDLE_NS`），并且 L2 空间压力大，可回写 L3（主机 DRAM）并释放 L2 空间。
5. **Migration atomicity**：迁移过程须设置 `state = MIGRATING_*`，对外查询以 CAS 检查该状态并等待或 retry；目录版本号需在迁移完成后更新以避免旧数据被错误读取。

---

##### Promote/Demote 伪代码（简化版）

```c
// Called periodically or triggered by access counters
function maintenance_worker():
    for each token_id in sampled_token_set:
        entry = read_directory(token_id)
        // promote logic
        if entry.layer == L3 and entry.freq >= PROMOTE_TO_L2_THRESHOLD:
            migrate_L3_to_L2(token_id)
        if entry.layer == L2 and entry.freq >= PROMOTE_TO_L1_THRESHOLD:
            migrate_L2_to_L1(token_id)
        // demote logic based on idle time or space pressure
        if entry.layer == L1 and (now() - entry.last_access_ts) > DEMOTE_FROM_L1_IDLE_NS:
            migrate_L1_to_L2(token_id)
        if entry.layer == L2 and (now() - entry.last_access_ts) > DEMOTE_FROM_L2_IDLE_NS:
            migrate_L2_to_L3(token_id)

// migration routines must be atomic w.r.t directory version
function migrate_L2_to_L1(token_id):
    if CAS(&directory[token_id].state, L2, MIGRATING_TO_L1):
        data = dma_read_from_L2(token_offset)
        if L1_is_full():
            victim = select_L1_victim() // LFU/LRU hybrid
            evict_L1_entry(victim)
        install_to_L1(token_id, data)
        directory[token_id].layer = L1
        directory[token_id].state = L1
```

---

###### L1 替换策略（LFU + LRU 混合）

由于 L1 容量小且访问模式既有长期热点也有短期热点，推荐使用 **LFU（频率）优先，LRU 作为 tie-breaker** 的混合策略。实现上可采用近似计数器 + aging：

* 每隔 `AGING_INTERVAL` 把所有 `freq` 右移一位（aging），避免旧热点永久占位。
* 选取最低 `score = freq + alpha * recency_rank` 的条目作为 victim（alpha 控制 recency 权重）。

---

###### Eviction 写回策略（Write-back / Write-through）

* 对于允许延迟一致性的字段，使用 **write-back**：在 L1 修改后仅设置 `dirty`，在 demote 时才将差分写回 L2。优点是减少写流量；缺点是增加失效风险（需持久化或在故障时处理）。
* 对于强一致字段（例如权限、key），采用 **write-through**：同步更新 L2 以确保主机/设备间一致。

---

###### 竞态与原子性处理（关键工程细节）

* 迁移与访问必须用 **版本号 + CAS** 保证：读取目录（version v1）→ 若 state == normal，继续；在迁移完成后原子地将 version++ 与 layer 修改为新值。读者或进程在发现 version 变化时必须 retry。
* 对被 `pinned` 的 token（如 admin/persistent QP）禁止迁移。
* 在迁移过程中对外读写请求有两种选择：

  1. **阻塞等待**：请求等待迁移完成（简单但增加延迟）
  2. **乐观并行**：请求触发读回或先读取旧副本并在完成后 reconcile（复杂但可提高并发）

论文中应说明两种选择的权衡，并建议实验评估两种策略下的 P99 延迟差异。

---

###### 统计、监控与自适应调节

为保证迁移策略在不同负载下表现稳定，建议实现以下监控与自适应机制：

* 实时统计：L1/L2 命中率、迁移速率、平均迁移延迟、DMA 带宽占用。
* 自适应阈值：若 DMA 带宽接近饱和且 L2 命中率低，则提升 `PROMOTE_TO_L1_THRESHOLD`（降低 promote 频率）；反之，若 L1 命中率低且主机 CPU 占用高，则降低阈值以加快 promote。
* 在论文中给出自适应策略的伪代码并在实验里对比固定阈值与自适应阈值的结果。

---

######  promote/demote 流程图（文字版）

1. **访问 → L1 hit**：处理完成，freq++。
2. **访问 → L1 miss, L2 hit**：schedule prefetch/直接 DMA 小块到 L1，处理后根据 freq 决定是否 promote。
3. **访问 → L1 miss, L2 miss**：发起 expand 到 host（blocking 或 async），或直接由 driver pre-stage 至 L2；在成功后按步骤 2 处理。
4. **资源压力**：当 L1 饱和并且出现 hotter entries，则选 victim evict（write-back 或 write-through），更新目录。

---

#### 可视化界面与用户系统

最终形成一个可视化界面，需要最基础的，登陆功能，模拟仿真功能，日志查询功能，数据处理功能