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
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rdma-cache-secret-key-change-in-production'
# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'results')
app.config['LOG_FOLDER'] = os.path.join(BASE_DIR, 'logs')
app.config['EXP_FOLDER'] = os.path.join(BASE_DIR, 'results')  # CSV/JSON 统一放这里

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

# ========== Migration stats (latest) ==========
@app.route('/api/migration/latest')
@login_required
def api_migration_latest():
    try:
        # 寻找 results 目录下最新的 JSON 结果（rdma_cache_sim 导出的）
        exp_dir = app.config['EXP_FOLDER']
        files = sorted(glob.glob(os.path.join(exp_dir, '*.json')), key=os.path.getmtime, reverse=True)
        if not files:
            return jsonify({'error': 'no result files'}), 404
        # 逐个查找含有迁移字段的结果
        fields = ['promote_l3_to_l2', 'promote_l2_to_l1', 'demote_l1_to_l2', 'demote_l2_to_l3']
        for fp in files:
            try:
                with open(fp, 'r') as f:
                    data = json.load(f)
                if all(k in data for k in fields):
                    return jsonify({
                        'file': os.path.basename(fp),
                        'promote_l3_to_l2': data.get('promote_l3_to_l2', 0),
                        'promote_l2_to_l1': data.get('promote_l2_to_l1', 0),
                        'demote_l1_to_l2': data.get('demote_l1_to_l2', 0),
                        'demote_l2_to_l3': data.get('demote_l2_to_l3', 0)
                    })
            except Exception:
                continue
        return jsonify({'error': 'no migration stats found in results'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# ========== A1 实验：压缩比与序列化开销 ==========
@app.route('/exp_a1')
@login_required
def exp_a1():
    return render_template('exp_a1.html')

@app.route('/api/exp_a1/run', methods=['POST'])
@login_required
def api_exp_a1_run():
    ctype = request.form.get('type', 'QP')
    N_str = request.form.get('N_values', '1000,10000,100000,1000000')
    N_values = [int(x.strip()) for x in N_str.split(',') if x.strip()]
    iters = int(request.form.get('iters', 10))
    random_fields = request.form.get('random_fields', 'true') == 'true'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(app.config['EXP_FOLDER'], f'exp_a1_{timestamp}.json')

    def run_exp():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exe = os.path.join(base_dir, 'build', 'exp_a1_compress')
        cmd = [exe, '--type', ctype, '--iters', str(iters), '--output', out_file, '--N_values', N_str]
        if not random_fields:
            cmd.append('--typical')
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=base_dir)
            if result.returncode != 0:
                with open(out_file, 'w') as f:
                    json.dump({'error': f'execution failed: {result.stderr}'}, f)
        except Exception as e:
            with open(out_file, 'w') as f:
                json.dump({'error': str(e)}, f)

    t = threading.Thread(target=run_exp)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started', 'file': out_file})

@app.route('/api/exp_a1/files')
@login_required
def api_exp_a1_files():
    files = []
    exp_folder = app.config['EXP_FOLDER']
    try:
        if not os.path.exists(exp_folder):
            app.logger.warning(f'EXP_FOLDER does not exist: {exp_folder}')
            return jsonify({'error': f'Folder not found: {exp_folder}'}), 500
        for f in os.listdir(exp_folder):
            if f.startswith('exp_a1_') and f.endswith('.json'):
                fp = os.path.join(exp_folder, f)
                if os.path.isfile(fp):
                    files.append({'filename': f, 'time': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
        files.sort(key=lambda x: x['time'], reverse=True)
    except Exception as e:
        app.logger.error(f'Error listing exp_a1 files: {e}')
        return jsonify({'error': str(e)}), 500
    return jsonify(files)

@app.route('/api/exp_a1/result')
@login_required
def api_exp_a1_result():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'file required'}), 400
    # 防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'invalid filename'}), 400
    fp = os.path.join(app.config['EXP_FOLDER'], filename)
    if not os.path.exists(fp):
        app.logger.warning(f'File not found: {fp}')
        return jsonify({'error': 'not found'}), 404
    try:
        with open(fp, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError as e:
        app.logger.error(f'JSON decode error for {fp}: {e}')
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f'Error reading file {fp}: {e}')
        return jsonify({'error': str(e)}), 500

# ========== B1 实验：基线延迟与吞吐对比 ==========
@app.route('/exp_b1')
@login_required
def exp_b1():
    return render_template('exp_b1.html')

@app.route('/api/exp_b1/run', methods=['POST'])
@login_required
def api_exp_b1_run():
    msg_size = request.form.get('msg_size', '256')
    qp_count = request.form.get('qp_count', '16')
    thread_count = request.form.get('thread_count', '8')
    ops_per_test = request.form.get('ops_per_test', '100000')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(app.config['EXP_FOLDER'], f'exp_b1_{timestamp}.json')

    def run_exp():
        # 使用BASE_DIR确保路径正确
        base_dir = BASE_DIR
        exe = os.path.join(base_dir, 'build', 'exp_b1_baseline')
        
        # 验证可执行文件是否存在
        if not os.path.exists(exe):
            error_msg = f'Executable not found: {exe}. Please run ./build.sh first.'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
            return
        
        # 确保可执行文件有执行权限
        if not os.access(exe, os.X_OK):
            error_msg = f'Executable not executable: {exe}'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
            return
        
        # 注意：exp_b1_baseline 会运行所有参数组合，所以这里只传递输出文件
        cmd = [exe, out_file]
        try:
            app.logger.info(f'Running exp_b1: {cmd} in {base_dir}')
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=base_dir)
            if result.returncode != 0:
                error_msg = f'Execution failed (code {result.returncode}): {result.stderr}'
                app.logger.error(error_msg)
                with open(out_file, 'w') as f:
                    json.dump({'error': error_msg, 'stdout': result.stdout, 'stderr': result.stderr}, f)
            else:
                app.logger.info(f'exp_b1 completed successfully, output: {out_file}')
        except FileNotFoundError as e:
            error_msg = f'File not found: {e}'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
        except Exception as e:
            error_msg = f'Exception: {str(e)}'
            app.logger.error(error_msg, exc_info=True)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)

    t = threading.Thread(target=run_exp)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started', 'file': out_file})

@app.route('/api/exp_b1/files')
@login_required
def api_exp_b1_files():
    files = []
    exp_folder = app.config['EXP_FOLDER']
    try:
        if not os.path.exists(exp_folder):
            app.logger.warning(f'EXP_FOLDER does not exist: {exp_folder}')
            return jsonify({'error': f'Folder not found: {exp_folder}'}), 500
        for f in os.listdir(exp_folder):
            if f.startswith('exp_b1_') and f.endswith('.json'):
                fp = os.path.join(exp_folder, f)
                if os.path.isfile(fp):
                    files.append({'filename': f, 'time': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
        files.sort(key=lambda x: x['time'], reverse=True)
    except Exception as e:
        app.logger.error(f'Error listing exp_b1 files: {e}')
        return jsonify({'error': str(e)}), 500
    return jsonify(files)

@app.route('/api/exp_b1/result')
@login_required
def api_exp_b1_result():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'file required'}), 400
    # 防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'invalid filename'}), 400
    fp = os.path.join(app.config['EXP_FOLDER'], filename)
    if not os.path.exists(fp):
        app.logger.warning(f'File not found: {fp}')
        return jsonify({'error': 'not found'}), 404
    try:
        with open(fp, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError as e:
        app.logger.error(f'JSON decode error for {fp}: {e}')
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f'Error reading file {fp}: {e}')
        return jsonify({'error': str(e)}), 500

# ========== B2 实验：Hot Inline 路径并发竞争测试 ==========
@app.route('/exp_b2')
@login_required
def exp_b2():
    return render_template('exp_b2.html')

@app.route('/api/exp_b2/run', methods=['POST'])
@login_required
def api_exp_b2_run():
    hot_token_count = request.form.get('hot_token_count', '10')
    ops_per_test = request.form.get('ops_per_test', '1000000')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(app.config['EXP_FOLDER'], f'exp_b2_{timestamp}.json')

    def run_exp():
        # 使用BASE_DIR确保路径正确
        base_dir = BASE_DIR
        exe = os.path.join(base_dir, 'build', 'exp_b2_hot_inline')
        
        # 验证可执行文件是否存在
        if not os.path.exists(exe):
            error_msg = f'Executable not found: {exe}. Please run ./build.sh first.'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
            return
        
        # 确保可执行文件有执行权限
        if not os.access(exe, os.X_OK):
            error_msg = f'Executable not executable: {exe}'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
            return
        
        # exp_b2_hot_inline 会运行所有并发度组合，所以这里只传递输出文件
        cmd = [exe, out_file]
        try:
            app.logger.info(f'Running exp_b2: {cmd} in {base_dir}')
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=base_dir)
            if result.returncode != 0:
                error_msg = f'Execution failed (code {result.returncode}): {result.stderr}'
                app.logger.error(error_msg)
                with open(out_file, 'w') as f:
                    json.dump({'error': error_msg, 'stdout': result.stdout, 'stderr': result.stderr}, f)
            else:
                app.logger.info(f'exp_b2 completed successfully, output: {out_file}')
        except FileNotFoundError as e:
            error_msg = f'File not found: {e}'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
        except Exception as e:
            error_msg = f'Exception: {str(e)}'
            app.logger.error(error_msg, exc_info=True)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)

    t = threading.Thread(target=run_exp)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started', 'file': out_file})

@app.route('/api/exp_b2/files')
@login_required
def api_exp_b2_files():
    files = []
    exp_folder = app.config['EXP_FOLDER']
    try:
        if not os.path.exists(exp_folder):
            app.logger.warning(f'EXP_FOLDER does not exist: {exp_folder}')
            return jsonify({'error': f'Folder not found: {exp_folder}'}), 500
        for f in os.listdir(exp_folder):
            if f.startswith('exp_b2_') and f.endswith('.json'):
                fp = os.path.join(exp_folder, f)
                if os.path.isfile(fp):
                    files.append({'filename': f, 'time': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
        files.sort(key=lambda x: x['time'], reverse=True)
    except Exception as e:
        app.logger.error(f'Error listing exp_b2 files: {e}')
        return jsonify({'error': str(e)}), 500
    return jsonify(files)

@app.route('/api/exp_b2/result')
@login_required
def api_exp_b2_result():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'file required'}), 400
    # 防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'invalid filename'}), 400
    fp = os.path.join(app.config['EXP_FOLDER'], filename)
    if not os.path.exists(fp):
        app.logger.warning(f'File not found: {fp}')
        return jsonify({'error': 'not found'}), 404
    try:
        with open(fp, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError as e:
        app.logger.error(f'JSON decode error for {fp}: {e}')
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f'Error reading file {fp}: {e}')
        return jsonify({'error': str(e)}), 500

# ========== B3 实验：Prefetched Batch batch size 与聚类策略影响 ==========
@app.route('/exp_b3')
@login_required
def exp_b3():
    return render_template('exp_b3.html')

@app.route('/api/exp_b3/run', methods=['POST'])
@login_required
def api_exp_b3_run():
    token_count = request.form.get('token_count', '10000')
    test_duration_sec = request.form.get('test_duration_sec', '60')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(app.config['EXP_FOLDER'], f'exp_b3_{timestamp}.json')

    def run_exp():
        # 使用BASE_DIR确保路径正确
        base_dir = BASE_DIR
        exe = os.path.join(base_dir, 'build', 'exp_b3_prefetched_batch')
        
        # 验证可执行文件是否存在
        if not os.path.exists(exe):
            error_msg = f'Executable not found: {exe}. Please run ./build.sh first.'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
            return
        
        # 确保可执行文件有执行权限
        if not os.access(exe, os.X_OK):
            error_msg = f'Executable not executable: {exe}'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
            return
        
        # exp_b3_prefetched_batch 会运行所有参数组合，所以这里只传递输出文件
        cmd = [exe, out_file]
        try:
            app.logger.info(f'Running exp_b3: {cmd} in {base_dir}')
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=base_dir)
            if result.returncode != 0:
                error_msg = f'Execution failed (code {result.returncode}): {result.stderr}'
                app.logger.error(error_msg)
                with open(out_file, 'w') as f:
                    json.dump({'error': error_msg, 'stdout': result.stdout, 'stderr': result.stderr}, f)
            else:
                app.logger.info(f'exp_b3 completed successfully, output: {out_file}')
        except FileNotFoundError as e:
            error_msg = f'File not found: {e}'
            app.logger.error(error_msg)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)
        except Exception as e:
            error_msg = f'Exception: {str(e)}'
            app.logger.error(error_msg, exc_info=True)
            with open(out_file, 'w') as f:
                json.dump({'error': error_msg}, f)

    t = threading.Thread(target=run_exp)
    t.daemon = True
    t.start()
    return jsonify({'status': 'started', 'file': out_file})

@app.route('/api/exp_b3/files')
@login_required
def api_exp_b3_files():
    files = []
    exp_folder = app.config['EXP_FOLDER']
    try:
        if not os.path.exists(exp_folder):
            app.logger.warning(f'EXP_FOLDER does not exist: {exp_folder}')
            return jsonify({'error': f'Folder not found: {exp_folder}'}), 500
        for f in os.listdir(exp_folder):
            if f.startswith('exp_b3_') and f.endswith('.json'):
                fp = os.path.join(exp_folder, f)
                if os.path.isfile(fp):
                    files.append({'filename': f, 'time': datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
        files.sort(key=lambda x: x['time'], reverse=True)
    except Exception as e:
        app.logger.error(f'Error listing exp_b3 files: {e}')
        return jsonify({'error': str(e)}), 500
    return jsonify(files)

@app.route('/api/exp_b3/result')
@login_required
def api_exp_b3_result():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'file required'}), 400
    # 防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'invalid filename'}), 400
    fp = os.path.join(app.config['EXP_FOLDER'], filename)
    if not os.path.exists(fp):
        app.logger.warning(f'File not found: {fp}')
        return jsonify({'error': 'not found'}), 404
    try:
        with open(fp, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError as e:
        app.logger.error(f'JSON decode error for {fp}: {e}')
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f'Error reading file {fp}: {e}')
        return jsonify({'error': str(e)}), 500

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

