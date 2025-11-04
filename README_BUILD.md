# RDMA缓存优化及仿真系统 - 构建和运行指南

## 系统要求

- C++17 编译器（g++ 7.0+ 或 clang++ 6.0+）
- CMake 3.15+
- Python 3.8+
- 现代Linux系统

## 构建步骤

### 1. 构建C++仿真程序

```bash
chmod +x build.sh
./build.sh
```

或手动构建：

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

编译成功后，可执行文件位于 `build/rdma_cache_sim`

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 3. 运行Web界面

```bash
chmod +x run.sh
./run.sh
```

或手动运行：

```bash
cd web
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py
```

Web界面将在 `http://localhost:5000` 启动

## 使用说明

### 登录系统

默认账户：
- 用户名: `admin`, 密码: `admin123`
- 用户名: `user`, 密码: `user123`

### 运行仿真

1. 登录系统后，进入"模拟仿真"页面
2. 配置仿真参数：
   - Token数量：模拟的对象总数
   - 访问次数：仿真的总访问次数
   - 工作负载类型：选择访问模式
   - 缓存容量：设置各层缓存大小
   - 迁移阈值：设置promote/demote的阈值
3. 点击"开始仿真"
4. 仿真在后台运行，完成后可查看结果

### 查看结果

1. 进入"结果查询"页面
2. 点击结果文件的"查看详情"按钮
3. 查看性能指标和统计数据

### 查看日志

1. 进入"日志查询"页面
2. 点击日志文件的"查看"按钮
3. 查看仿真运行的详细日志

### 数据分析

1. 进入"数据处理"页面
2. 查看可视化图表：
   - 命中率历史趋势
   - 缓存层命中率分布
   - 迁移操作统计

## 命令行运行仿真

也可以直接运行C++仿真程序：

```bash
./build/rdma_cache_sim [tokens] [accesses] [workload_type] [output_file] [log_file]
```

参数说明：
- `tokens`: Token数量（默认10000）
- `accesses`: 访问次数（默认100000）
- `workload_type`: 工作负载类型（0=均匀, 1=Zipfian, 2=顺序, 3=随机游走）
- `output_file`: 结果JSON文件（默认results.json）
- `log_file`: 日志文件（默认simulation.log）

示例：
```bash
./build/rdma_cache_sim 10000 100000 1 results.json simulation.log
```

## 目录结构

```
rdma-cache-sys/
├── include/          # C++头文件
├── src/              # C++源文件
├── web/              # Python Web界面
│   ├── app.py        # Flask应用
│   ├── templates/    # HTML模板
│   └── static/       # 静态资源
├── build/            # 构建目录
├── results/          # 仿真结果（JSON）
├── logs/             # 仿真日志
├── CMakeLists.txt    # CMake配置
├── requirements.txt  # Python依赖
└── README.md         # 项目说明
```

## 故障排除

### 编译错误

- 确保使用C++17编译器
- 检查CMake版本 >= 3.15

### Web界面无法启动

- 检查Python版本 >= 3.8
- 确保所有依赖已安装：`pip install -r requirements.txt`
- 检查端口5000是否被占用

### 仿真程序找不到

- 确保已运行 `./build.sh` 构建程序
- 检查 `build/rdma_cache_sim` 是否存在且有执行权限

## 性能调优

根据实际硬件和需求，可以调整以下参数：

- L1缓存容量：影响热数据缓存大小
- L2/L3缓存容量：影响总体缓存能力
- Promote阈值：影响数据迁移频率
- 工作负载类型：影响访问模式

建议进行多组实验以找到最佳配置。

