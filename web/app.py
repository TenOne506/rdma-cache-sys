"""
RDMA Cache Simulation Web Interface
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json
import os
import subprocess
import threading
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rdma-cache-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'results'
app.config['LOG_FOLDER'] = 'logs'
app.config['EXP_FOLDER'] = 'results'  # CSV/JSON 统一放这里

# Flask-Login配置
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

# 简单用户管理（生产环境应使用数据库）
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# 用户数据库（简化实现）
users = {
    'admin': {'password': generate_password_hash('admin123'), 'id': 'admin'},
    'user': {'password': generate_password_hash('user123'), 'id': 'user'}
}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
@login_required
def index():
    return render_template('index.html')

# 拓扑图页面与数据接口
@app.route('/topology')
@login_required
def topology_page():
    return render_template('topology.html')

@app.route('/api/topology/summary')
@login_required
def api_topology_summary():
    # 取UPLOAD_FOLDER下最新的结果JSON
    latest = None
    latest_mtime = 0
    base = app.config['UPLOAD_FOLDER']
    if os.path.exists(base):
        for f in os.listdir(base):
            if f.endswith('.json'):
                fp = os.path.join(base, f)
                m = os.path.getmtime(fp)
                if m > latest_mtime:
                    latest_mtime = m
                    latest = fp
    if not latest:
        return jsonify({'error': 'no results'}), 404
    with open(latest, 'r') as f:
        data = json.load(f)
    # 提取需要的指标
    summary = {
        'l1_share': data.get('l1_share', data.get('l1_hit_rate', 0)),
        'l2_share': data.get('l2_share', data.get('l2_hit_rate', 0)),
        'l3_share': data.get('l3_share', data.get('l3_hit_rate', 0)),
        'avg_latency_ns': data.get('avg_latency_ns', 0),
        'promote_l3_to_l2': data.get('promote_l3_to_l2', 0),
        'promote_l2_to_l1': data.get('promote_l2_to_l1', 0),
        'demote_l1_to_l2': data.get('demote_l1_to_l2', 0),
        'demote_l2_to_l3': data.get('demote_l2_to_l3', 0),
        'file': os.path.basename(latest)
    }
    return jsonify(summary)

# ========== 压缩基准 ==========
@app.route('/compress')
@login_required
def compress_page():
    return render_template('compress.html')

@app.route('/api/compress/run', methods=['POST'])
@login_required
def api_compress_run():
    ctype = request.form.get('type', 'QP')
    N = int(request.form.get('N', 100000))
    iters = int(request.form.get('iters', 5))
    threads = int(request.form.get('threads', 4))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(app.config['EXP_FOLDER'], f'bench_{timestamp}.json')

    def run_bench():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exe = os.path.join(base_dir, 'build', 'bench_token_compress')
        cmd = [exe, '--type', ctype, '--N', str(N), '--iters', str(iters), '--threads', str(threads)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            stdout = result.stdout
            total_bytes = 0
            avg_us = 0.0
            throughput = 0.0
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith('total_bytes='):
                    total_bytes = int(line.split('=')[1])
                elif line.startswith('avg_us_per_encode_decode='):
                    avg_us = float(line.split('=')[1])
                elif line.startswith('throughput_tokens_per_sec='):
                    throughput = float(line.split('=')[1])
            payload = {
                'type': ctype,
                'N': N,
                'iters': iters,
                'threads': threads,
                'total_bytes': total_bytes,
                'avg_us_per_encode_decode': avg_us,
                'throughput_tokens_per_sec': throughput,
                'timestamp': timestamp
            }
            with open(out_file, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            with open(out_file, 'w') as f:
                json.dump({'error': str(e)}, f)

    t = threading.Thread(target=run_bench)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started', 'file': out_file})

@app.route('/api/compress/files')
@login_required
def api_compress_files():
    files = []
    for f in os.listdir(app.config['EXP_FOLDER']):
        if f.startswith('bench_') and f.endswith('.json'):
            fp = os.path.join(app.config['EXP_FOLDER'], f)
            files.append({'filename': f, 'time': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
    files.sort(key=lambda x: x['time'], reverse=True)
    return jsonify(files)

@app.route('/api/compress/result')
@login_required
def api_compress_result():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'file required'}), 400
    fp = os.path.join(app.config['EXP_FOLDER'], filename)
    if not os.path.exists(fp):
        return jsonify({'error': 'not found'}), 404
    with open(fp, 'r') as f:
        data = json.load(f)
    return jsonify(data)

# ========== 仿真 ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and check_password_hash(users[username]['password'], password):
            user = User(users[username]['id'])
            login_user(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/simulation', methods=['GET', 'POST'])
@login_required
def simulation():
    if request.method == 'POST':
        # 获取配置参数
        config = {
            'num_tokens': int(request.form.get('num_tokens', 10000)),
            'num_accesses': int(request.form.get('num_accesses', 100000)),
            'workload_type': int(request.form.get('workload_type', 1)),
            'l1_capacity': int(request.form.get('l1_capacity', 8192)),
            'l2_capacity_mb': int(request.form.get('l2_capacity_mb', 1024)),
            'l3_capacity_mb': int(request.form.get('l3_capacity_mb', 4096)),
            'promote_l1_threshold': int(request.form.get('promote_l1_threshold', 128)),
            'promote_l2_threshold': int(request.form.get('promote_l2_threshold', 16)),
        }
        
        # 生成唯一的结果文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{timestamp}.json')
        log_file = os.path.join(app.config['LOG_FOLDER'], f'sim_{timestamp}.log')
        
        # 在后台线程中运行仿真
        thread = threading.Thread(
            target=run_simulation,
            args=(config, result_file, log_file)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'result_file': result_file,
            'log_file': log_file
        })
    
    return render_template('simulation.html')

def run_simulation(config, result_file, log_file):
    """运行C++仿真程序"""
    try:
        # 获取项目根目录
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_path = os.path.join(base_dir, 'build', 'rdma_cache_sim')
        
        # 构建命令
        cmd = [
            sim_path,
            str(config['num_tokens']),
            str(config['num_accesses']),
            str(config['workload_type']),
            result_file,
            log_file
        ]
        
        # 运行仿真
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode != 0:
            # 记录错误
            with open(log_file, 'a') as f:
                f.write(f"Error: {result.stderr}\n")
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Exception: {str(e)}\n")

@app.route('/experiments', methods=['GET'])
@login_required
def experiments_page():
    return render_template('experiments.html')

@app.route('/api/experiments/run', methods=['POST'])
@login_required
def api_experiments_run():
    tokens = int(request.form.get('tokens', 5000))
    accesses = int(request.form.get('accesses', 20000))
    workload = int(request.form.get('workload', 1))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(app.config['EXP_FOLDER'], f'exp_{timestamp}.csv')
    
    def run_job():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exe = os.path.join(base_dir, 'build', 'cache_experiments')
        cmd = [exe, '-c', csv_file, '-t', str(tokens), '-a', str(accesses), '-w', str(workload)]
        try:
            subprocess.run(cmd, check=True, timeout=3600)
        except subprocess.CalledProcessError as e:
            pass
    
    t = threading.Thread(target=run_job)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started', 'csv': csv_file})

@app.route('/api/experiments/files')
@login_required
def api_experiments_files():
    files = []
    for f in os.listdir(app.config['EXP_FOLDER']):
        if f.endswith('.csv'):
            fp = os.path.join(app.config['EXP_FOLDER'], f)
            files.append({'filename': f, 'time': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
    files.sort(key=lambda x: x['time'], reverse=True)
    return jsonify(files)

@app.route('/api/experiments/data')
@login_required
def api_experiments_data():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'file required'}), 400
    fp = os.path.join(app.config['EXP_FOLDER'], filename)
    if not os.path.exists(fp):
        return jsonify({'error': 'not found'}), 404
    try:
        df = pd.read_csv(fp)
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/experiments/l3', methods=['GET'])
@login_required
def experiments_l3_page():
    return render_template('experiments_l3.html')

@app.route('/api/experiments_l3/run', methods=['POST'])
@login_required
def api_experiments_l3_run():
    tokens = int(request.form.get('tokens', 5000))
    accesses = int(request.form.get('accesses', 20000))
    workload = int(request.form.get('workload', 1))
    pre_ratio = float(request.form.get('pre_ratio', 0.0))
    expand_window = int(request.form.get('expand_window', 1))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(app.config['EXP_FOLDER'], f'exp_l3_{timestamp}.csv')
    
    def run_job():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exe = os.path.join(base_dir, 'build', 'cache_experiments')
        cmd = [exe, '-c', csv_file, '-t', str(tokens), '-a', str(accesses), '-w', str(workload), '-r', str(pre_ratio), '-e', str(expand_window)]
        try:
            subprocess.run(cmd, check=True, timeout=3600)
        except subprocess.CalledProcessError:
            pass
    
    t = threading.Thread(target=run_job)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started', 'csv': csv_file})

@app.route('/api/experiments_l3/files')
@login_required
def api_experiments_l3_files():
    files = []
    for f in os.listdir(app.config['EXP_FOLDER']):
        if f.startswith('exp_l3_') and f.endswith('.csv'):
            fp = os.path.join(app.config['EXP_FOLDER'], f)
            files.append({'filename': f, 'time': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
    files.sort(key=lambda x: x['time'], reverse=True)
    return jsonify(files)

@app.route('/api/experiments_l3/data')
@login_required
def api_experiments_l3_data():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'file required'}), 400
    fp = os.path.join(app.config['EXP_FOLDER'], filename)
    if not os.path.exists(fp):
        return jsonify({'error': 'not found'}), 404
    try:
        df = pd.read_csv(fp)
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results')
@login_required
def results():
    """显示所有结果文件"""
    results = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith('.json'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                mtime = os.path.getmtime(filepath)
                results.append({
                    'filename': filename,
                    'time': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    results.sort(key=lambda x: x['time'], reverse=True)
    return render_template('results.html', results=results)

@app.route('/results/<filename>')
@login_required
def view_result(filename):
    """查看特定结果文件"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return jsonify(data)

@app.route('/logs')
@login_required
def logs():
    """显示日志文件列表"""
    logs = []
    if os.path.exists(app.config['LOG_FOLDER']):
        for filename in os.listdir(app.config['LOG_FOLDER']):
            if filename.endswith('.log'):
                filepath = os.path.join(app.config['LOG_FOLDER'], filename)
                mtime = os.path.getmtime(filepath)
                logs.append({
                    'filename': filename,
                    'time': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'size': os.path.getsize(filepath)
                })
    
    logs.sort(key=lambda x: x['time'], reverse=True)
    return render_template('logs.html', logs=logs)

@app.route('/logs/<filename>')
@login_required
def view_log(filename):
    """查看特定日志文件"""
    filepath = os.path.join(app.config['LOG_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    return render_template('view_log.html', filename=filename, content=content)

@app.route('/data')
@login_required
def data():
    """数据处理和可视化页面"""
    return render_template('data.html')

@app.route('/api/data/statistics')
@login_required
def api_statistics():
    """获取统计数据的API"""
    stats = {
        'total_results': 0,
        'total_logs': 0,
        'avg_hit_rate': 0.0
    }
    
    # 统计结果文件
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        json_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.json')]
        stats['total_results'] = len(json_files)
        
        # 计算平均命中率
        hit_rates = []
        for filename in json_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if 'overall_hit_rate' in data:
                        hit_rates.append(data['overall_hit_rate'])
            except:
                pass
        
        if hit_rates:
            stats['avg_hit_rate'] = sum(hit_rates) / len(hit_rates)
    
    # 统计日志文件
    if os.path.exists(app.config['LOG_FOLDER']):
        log_files = [f for f in os.listdir(app.config['LOG_FOLDER']) if f.endswith('.log')]
        stats['total_logs'] = len(log_files)
    
    return jsonify(stats)

@app.route('/api/data/history')
@login_required
def api_history():
    """获取历史数据用于绘图"""
    history = []
    
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            if filename.endswith('.json'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        mtime = os.path.getmtime(filepath)
                        history.append({
                            'time': datetime.fromtimestamp(mtime).isoformat(),
                            'l1_hit_rate': data.get('l1_hit_rate', 0),
                            'l2_hit_rate': data.get('l2_hit_rate', 0),
                            'l3_hit_rate': data.get('l3_hit_rate', 0),
                            'overall_hit_rate': data.get('overall_hit_rate', 0)
                        })
                except:
                    pass
    
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

