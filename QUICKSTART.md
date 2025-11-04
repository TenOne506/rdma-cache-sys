# 快速开始指南

## 第一步：构建C++仿真程序

```bash
./build.sh
```

这将编译C++仿真核心，生成可执行文件 `build/rdma_cache_sim`

## 第二步：启动Web界面

```bash
./run.sh
```

或手动启动：

```bash
cd web
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python app.py
```

## 第三步：访问系统

1. 打开浏览器访问：`http://localhost:5000`
2. 使用默认账户登录：
   - 用户名：`admin`
   - 密码：`admin123`

## 第四步：运行仿真

1. 点击"模拟仿真"菜单
2. 配置仿真参数（可使用默认值）
3. 点击"开始仿真"
4. 等待仿真完成（后台运行）
5. 查看"结果查询"页面查看结果

## 命令行直接运行仿真

```bash
./build/rdma_cache_sim 10000 100000 1 results/test.json logs/test.log
```

参数：
- 10000: Token数量
- 100000: 访问次数
- 1: 工作负载类型（0=均匀, 1=Zipfian, 2=顺序, 3=随机游走）
- results/test.json: 结果文件
- logs/test.log: 日志文件

## 故障排除

### 编译失败
- 确保有C++17编译器：`g++ --version`
- 确保CMake已安装：`cmake --version`

### Python依赖问题
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 端口被占用
修改 `web/app.py` 中的端口号：
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # 改为其他端口
```

## 功能说明

### 1. 模拟仿真
- 配置仿真参数
- 选择工作负载类型
- 设置缓存容量和迁移阈值
- 启动后台仿真

### 2. 结果查询
- 查看所有仿真结果
- 查看详细的性能指标
- 查看命中率和迁移统计

### 3. 日志查询
- 查看仿真运行日志
- 调试和问题排查

### 4. 数据处理
- 可视化图表展示
- 历史趋势分析
- 统计信息汇总

## 实验设计建议

### 基准实验
1. **不同工作负载类型对比**
   - Uniform vs Zipfian vs Sequential vs Random Walk
   - 观察命中率差异

2. **缓存容量影响**
   - 调整L1/L2/L3容量
   - 观察性能变化

3. **迁移阈值优化**
   - 调整promote/demote阈值
   - 找到最佳平衡点

### 高级实验
1. **混合工作负载**
   - 组合不同访问模式

2. **动态阈值调整**
   - 自适应迁移策略

3. **不同Token类型对比**
   - PD/MR/CQ/QP的性能差异

## 下一步

- 查看 `README_BUILD.md` 了解详细构建说明
- 查看 `README.md` 了解系统架构和设计原理
- 修改代码进行自定义实验

