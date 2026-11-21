"""Read-only status source: follows a server the console did not start.

`python -m xtier.console --readonly` serves it. It probes the model's port (SGLang's /metrics and /get_server_info),
reads the server's log for its tail and reads nvidia-smi; the page has no model rail, no launch options and no
start / stop route, and nothing here can signal a process. Use it to watch a server started by hand (a run script, a
systemd unit) with the console's own HUD.
"""

import ast
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from . import core

# the metrics we take (labels other than tp_rank="0" would double-count a gauge)
_KEEP = (
    'sglang:num_running_reqs', 'sglang:num_queue_reqs', 'sglang:gen_throughput',
    'sglang:kv_used_tokens', 'sglang:max_total_num_tokens', 'sglang:cache_hit_rate',
    'sglang:cached_tokens_total', 'sglang:spec_accept_length', 'sglang:spec_accept_rate',
    'sglang:kv_cache_memory_usage_gb', 'sglang:num_requests_total',
    'sglang:prompt_tokens_total', 'sglang:generation_tokens_total',
    'sglang:hicache_host_used_tokens', 'sglang:hicache_host_total_tokens',
)
_CACHE_LINE = re.compile(r'^\[xtier\] XTier cache: \d+ windows, hit ([\d.]+), ([\d.]+) distinct experts and '
                         r'(\d+) us CPU per layer call; (\d+) swaps in (\d+) rounds[^+]*\+(\d+) of \+(\d+) slots', re.M)
# the server's start-up line: the whole configuration as a Python dict (the launch arguments)
_ARGS = re.compile(r'^\[[^\]]*\] server_args=(\{.*\})\s*$', re.M)
# the launch arguments the dashboard shows per model
_PARAM_KEYS = (
    'context_length', 'tp_size', 'max_running_requests', 'max_total_tokens', 'max_total_num_tokens',
    'chunked_prefill_size',
    'max_prefill_tokens', 'prefill_decode_interval', 'schedule_policy', 'mem_fraction_static',
    'prefill_max_requests', 'cuda_graph_max_bs_decode', 'max_mamba_cache_size', 'enable_hierarchical_cache',
    'hicache_ratio', 'speculative_algorithm', 'speculative_dspark_block_size', 'speculative_num_draft_tokens',
    'attention_backend', 'disable_custom_all_reduce', 'disable_radix_cache', 'model_path', 'served_model_name',
)
_PARAM_LABELS = {
    'context_length': '上下文', 'tp_size': 'TP', 'max_running_requests': '并发上限', 'max_total_tokens': 'KV 池',
    'max_total_num_tokens': 'KV 池',
    'chunked_prefill_size': 'chunk', 'max_prefill_tokens': 'prefill 上限', 'prefill_decode_interval': 'PDI',
    'schedule_policy': '调度', 'mem_fraction_static': 'mem 比例', 'prefill_max_requests': 'prefill 请求数',
    'cuda_graph_max_bs_decode': 'graph bs', 'max_mamba_cache_size': 'mamba 缓存', 'enable_hierarchical_cache': 'HiCache',
    'hicache_ratio': 'HiCache 比例', 'speculative_algorithm': '投机', 'speculative_dspark_block_size': 'DSpark 块',
    'speculative_num_draft_tokens': '草稿 token', 'attention_backend': 'attention', 'disable_custom_all_reduce': '关自定义 AR',
    'disable_radix_cache': '关前缀缓存', 'model_path': '权重', 'served_model_name': '模型名',
}


def _get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode('utf-8', 'replace')


def _parse(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        head, _, raw = line.rpartition(' ')
        if head.split('{', 1)[0] not in _KEEP:
            continue
        labels = head[head.find('{'):] if '{' in head else ''
        if 'tp_rank="' in labels and 'tp_rank="0"' not in labels:
            continue
        try:
            out[head.split('{', 1)[0]] = float(raw)
        except ValueError:
            pass
    return out


def _gpus(ids):
    """GPU0+2 only (ids, nvidia-smi -i): the other cards may run something else."""
    try:
        out = subprocess.run(['nvidia-smi', '-i', ids, '--query-gpu=name,memory.used,memory.total,utilization.gpu,'
                                                       'power.draw', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout
        rows = [[x.strip() for x in ln.split(',')] for ln in out.splitlines() if ln.count(',') >= 4]
        if not rows:
            return {}
        num = lambda s: core._number(s)  # noqa: E731 - the console's own parser
        return {'name': rows[0][0].replace('NVIDIA ', ''), 'count': len(rows),
                'mem_used': round(sum(num(r[1]) for r in rows) / 1024, 1),
                'mem_total': round(sum(num(r[2]) for r in rows) / 1024, 1),
                'util': round(sum(num(r[3]) for r in rows) / len(rows)),
                'power': round(sum(num(r[4]) for r in rows))}
    except (OSError, subprocess.TimeoutExpired):
        return {}


def server_elapsed(port):
    """Seconds since the launch_server process that serves this port started (its /proc entry's mtime), or None."""
    want = str(port).encode()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if not any(b'launch_server' in a for a in argv) or b'--port' not in argv:
            continue
        i = argv.index(b'--port')
        if i + 1 < len(argv) and argv[i + 1] == want:
            try:
                return max(0.0, time.time() - entry.stat().st_mtime)
            except OSError:
                return None
    return None


def newest_log():
    """The newest log that holds a launch_server line (skips the console's own, the guard's and the bench logs)."""
    cands = []
    for p in Path('/tmp/opencode').glob('xtier*.log'):
        if any(tag in p.name for tag in ('console', 'memguard', 'status-page')):
            continue
        try:
            cands.append((p.stat().st_mtime, p))
        except OSError:
            continue
    for _, p in sorted(cands, reverse=True):
        try:
            with open(p, 'rb') as fh:   # the tail carries the newest telemetry
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - 16384))
                tail = fh.read()
            if b'XTier cache:' in tail or b'Decode batch' in tail or b'launch_server' in tail:
                return str(p)
        except OSError:
            continue
    return str(cands[0][1]) if cands else None


def _gpus_of_tree(pid):
    """The GPU indexes a process runs on (nvidia-smi: works without reading another user's /proc).

    nvidia-smi lists the worker processes, not the launcher, so the whole process tree counts."""
    children = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            ppid = int(stat.rsplit(')', 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))
    tree = {pid}
    stack = [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in tree:
                tree.add(child)
                stack.append(child)
    uuid_to_index = {}
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5).stdout
        for line in q.splitlines():
            idx, _, uuid = line.partition(',')
            uuid_to_index[uuid.strip()] = idx.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''
    ids = set()
    for line in out.splitlines():
        p, _, uuid = line.partition(',')
        if p.strip().isdigit() and int(p.strip()) in tree:
            index = uuid_to_index.get(uuid.strip())
            if index is not None:
                ids.add(index)
    return ','.join(sorted(ids, key=lambda x: int(x) if x.isdigit() else 0))


def find_log_for_port(port):
    """The log file of the server on this port (another user's process hides its stdout from us).

    Looks in the usual places and takes the newest file whose head mentions the port."""
    cands = []
    for pattern in ('/var/log/llm/*.log', '/tmp/opencode/*.log', '/tmp/*.log'):
        cands.extend(Path('/').glob(pattern.lstrip('/')))
    for path in sorted(cands, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        try:
            if path.stat().st_size > 64 << 20:
                continue
            with open(path, 'rb') as fh:
                head = fh.read(4 << 20).decode('utf-8', 'replace')
            if f"'port': {port}" in head or f'--port {port}' in head or f'"port": {port}' in head:
                return str(path)
        except OSError:
            continue
    return None


def discover_servers():
    """Every XTier/SGLang model server on this host: its port, its GPUs and its log.

    Reads /proc only: the launch_server processes (their --port), their CUDA_VISIBLE_DEVICES and their stdout
    (the log file a service redirects to). A dashboard of several models at once needs no configuration.
    Another user's process hides its environment and its file descriptors from us; nvidia-smi then says which
    GPU it runs on, and the log falls back to the newest one that holds this port's server."""
    out = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if not any(b'launch_server' in a for a in argv):
            continue
        port = None
        if b'--port' in argv:
            i = argv.index(b'--port')
            if i + 1 < len(argv):
                try:
                    port = int(argv[i + 1])
                except ValueError:
                    port = None
        if port is None:
            continue
        gpus = ''
        try:
            for item in (entry / 'environ').read_bytes().split(b'\0'):
                if item.startswith(b'CUDA_VISIBLE_DEVICES='):
                    gpus = item.split(b'=', 1)[1].decode('utf-8', 'replace')
                    break
        except OSError:
            pass
        if not gpus:
            gpus = _gpus_of_tree(int(entry.name))
        log = None
        try:
            target = os.readlink(entry / 'fd' / '1')
            if target and not target.startswith(('pipe:', 'socket:', '/dev/', 'anon_inode:')):
                log = target
        except OSError:
            pass
        if not log:
            log = find_log_for_port(port)
        out.append({'pid': int(entry.name), 'port': port, 'gpus': gpus, 'log': log})
    return sorted(out, key=lambda s: s['port'])

# --- systemd: the enabled model services, listed once at start-up (no process / port scan) ---

_LLM_SIGN = re.compile(r'(launch_server|sglang|vllm|ninfer|ollama|llama-server|llama\.cpp|text-generation-launcher|'
                       r'serve/server\.py|trtllm|xinference|mindie)', re.I)
_NOT_LLM = re.compile(r'(dashboard|console|litellm|gateway|prometheus|grafana|redis|postgres)', re.I)
_UNIT_ENV = re.compile(r'^Environment=(?:"([^"]*)"|(\S+))\s*$', re.M)
_UNIT_OUT = re.compile(r'^Standard(?:Output|Error)=append:(\S+)\s*$', re.M)
_UNIT_DESC = re.compile(r'^Description=(.*)$', re.M)
_UNIT_FLAG_PORT = re.compile(r'--(?:api-)?port[= ]+(\d+)')
_UNIT_DEFAULT_PORT = re.compile(r'\bPORT=\$\{PORT:-(\d+)\}')
_UNIT_PLAIN_PORT = re.compile(r'^\s*(?:export\s+)?PORT=(\d+)\s*$', re.M)


def _systemctl(*args):
    try:
        p = subprocess.run(['systemctl', *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return p.stdout if p.returncode == 0 else ''


def _exec_line(text):
    """ExecStart= with systemd's line continuations; the last definition wins, as in systemd."""
    out, on = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('#'):
            continue
        if s.startswith('ExecStart='):
            if not on:
                out = []
            on = True
            s = s[len('ExecStart='):]
        if not on:
            continue
        if s.endswith('\\'):
            out.append(s[:-1].strip())
        else:
            out.append(s.strip())
            on = False
    return ' '.join(out)


def _wrapped_scripts(exec_line, depth=2):
    """The .sh files ExecStart hands off to (/bin/bash /opt/Xtier/run-....sh), followed depth levels."""
    out, todo, seen = [], re.findall(r'(/[^\s"\'\\]+\.sh)', exec_line), set()
    for _ in range(depth):
        nxt = []
        for path in todo:
            if path in seen:
                continue
            seen.add(path)
            try:
                with open(path, 'rb') as fh:
                    text = fh.read(256 << 10).decode('utf-8', 'replace')
            except OSError:
                continue
            out.append(text)
            nxt.extend(re.findall(r'(/[^\s"\'\\]+\.sh)', text))
        todo = nxt
        if not todo:
            break
    return out


def _strip_comments(text):
    return '\n'.join(ln for ln in text.splitlines() if not ln.lstrip().startswith(('#', ';')))


def _unit_env(text):
    env = {}
    for m in _UNIT_ENV.finditer(text):
        raw = m.group(1) or m.group(2) or ''
        key, sep, value = raw.partition('=')
        if sep:
            env[key] = value.strip().strip('"\'')
    return env


def _unit_port(unit_text, scripts, env):
    """PORT= in the unit first, then --port, then the wrapper's PORT=${PORT:-...} default."""
    for key in ('PORT', 'SERVER_PORT', 'API_PORT'):
        if env.get(key, '').isdigit():
            return int(env[key])
    for text in [unit_text, *scripts]:
        text = _strip_comments(text)
        m = _UNIT_FLAG_PORT.search(text)
        if m:
            return int(m.group(1))
        m = _UNIT_DEFAULT_PORT.search(text) or _UNIT_PLAIN_PORT.search(text)
        if m:
            return int(m.group(1))
    return None


def systemd_servers():
    """Every enabled systemd service that looks like a model server, with its port, GPUs and log.

    One pass over `systemctl list-unit-files --state=enabled`; nothing here watches processes or probes ports.
    A unit qualifies when its ExecStart (or the .sh it hands off to) launches a known serving binary; the port
    comes from Environment=PORT=..., --port, or the wrapper's PORT=${PORT:-...} default. Disabled units are not
    listed: enable the service and restart the dashboard to see it."""
    out = []
    listing = _systemctl('list-unit-files', '--type=service', '--state=enabled', '--no-legend', '--no-pager')
    for line in listing.splitlines():
        fields = line.split()
        unit = fields[0] if fields else ''
        if not unit.endswith('.service') or _NOT_LLM.search(unit):
            continue
        path = _systemctl('show', unit, '-p', 'FragmentPath', '--value').strip()
        if not path:
            continue
        try:
            with open(path, 'rb') as fh:
                text = fh.read(1 << 20).decode('utf-8', 'replace')
        except OSError:
            continue
        exec_line = _exec_line(text)
        scripts = _wrapped_scripts(exec_line)
        if not _LLM_SIGN.search(text + '\n' + '\n'.join(scripts)):
            continue
        env = _unit_env(text)
        port = _unit_port(text, scripts, env)
        if not port:
            continue
        log = _UNIT_OUT.search(text)
        desc = _UNIT_DESC.search(text)
        out.append({'unit': unit, 'port': port, 'gpus': env.get('CUDA_VISIBLE_DEVICES', ''),
                    'log': log.group(1) if log else None, 'desc': desc.group(1).strip() if desc else ''})
    return sorted(out, key=lambda s: (s['port'], s['unit']))


class Readonly:
    """Console.status() / console_data() for a server that is already running (no start, no stop, no options)."""

    def __init__(self, model, log=None, gpu_ids='0,2', spark=48):
        self.model = model
        self.id = model['id']
        self.log = log or newest_log()
        self.gpu_ids = gpu_ids
        self.spark_len = spark
        self._spark = []
        self._lock = threading.Lock()
        self._cache = {'t': 0.0, 'body': None}
        self._head = None
        self._started = time.time()

    def _head_info(self):
        """version, layer count and the launch arguments: the server logs them once at start-up, far above a log tail.
        Everything the HUD needs that is not a live metric comes from here, so nothing polls /get_server_info (on this
        build it can take seconds while the scheduler is busy, and it logs a deprecation line on every call)."""
        if self._head is None:
            self._head = {'version': '', 'layers': 0, 'args': {}}
            text = ''
            if self.log:
                try:
                    with open(self.log, 'rb') as fh:
                        text = fh.read(4 << 20).decode('utf-8', 'replace').replace('\r', '\n')
                except OSError:
                    text = ''
            version = core._VERSION.findall(text)
            args = {}
            # the log holds every instance that ran (it is appended to): the newest server_args is the one that
            # describes the server that runs now, not the first one in the file
            found = _ARGS.findall(text)
            if found:
                try:
                    value = ast.literal_eval(found[-1])
                    args = value if isinstance(value, dict) else {}
                except (ValueError, SyntaxError):
                    args = {}
            self._head = {'version': version[-1] if version else '',
                          'layers': len(set(core._LAYER.findall(text))), 'args': args}
        return self._head

    # --- what the page renders ---

    def console_data(self):
        return {'models': {self.id: {f: self.model[f] for f in core.PUBLIC if f in self.model}},
                'options': {self.id: []}, 'defaults': {self.id: {}}, 'saved': {'model': self.id},
                'readonly': True,
                'texts': {'subtitle': f"只读状态页 · 跟随 127.0.0.1:{self.model['port']} 上已在运行的服务"
                                      '（没有模型切换，也没有启停按钮）',
                          'foot': '只读页面：不含模型切换与启停接口；启动 / 停止请用 run-ds41-l20-tp2.sh。'}}

    def status(self):
        with self._lock:
            if self._cache['body'] is not None and time.monotonic() - self._cache['t'] < 1.2:
                return self._cache['body']
        body = self._status()
        with self._lock:
            self._cache.update(t=time.monotonic(), body=body)
        return body

    # --- probes ---

    def _status(self):
        m = self.model
        body = {'model': self.id, 'label': f"XTier · {m['name']}", 'port': m['port'], 'served': m['served'],
                'gpu': _gpus(self.gpu_ids),
                'tail': [t for t in (core.log_tail(self.log, 40) if self.log else []) if 'deprecated' not in t]}
        head = self._head_info()
        # the launch arguments are in the log from the first seconds: the parameters column shows them while loading too
        body['args'] = {k: head['args'][k] for k in _PARAM_KEYS if head['args'].get(k) is not None}
        if head['version']:
            body['version'] = head['version']
        if head['layers']:
            body['layers'] = head['layers']
        tail_text = core._read_tail(self.log) if self.log else ''
        elapsed = server_elapsed(m['port'])
        if elapsed is not None:
            body['elapsed'] = round(elapsed)
        try:
            metrics = _parse(_get(f"http://127.0.0.1:{m['port']}/metrics"))
        except Exception as exc:   # noqa: BLE001 - offline is the normal case here, the page says which
            body['error'] = f'{type(exc).__name__}: {exc}'[:200]
            if elapsed is None:
                body['state'] = 'offline'
            else:   # a process is there but the API is not up yet (or is going down)
                phase, label, pct = core._progress(tail_text)
                body.update(state='loading', progress=pct, phase=phase, phase_label=label)
            return body

        info = head['args']

        max_tok = metrics.get('sglang:max_total_num_tokens') or info.get('max_total_tokens') or 0
        used_tok = metrics.get('sglang:kv_used_tokens') or 0
        tps = round(metrics.get('sglang:gen_throughput', 0.0), 1)
        accept = metrics.get('sglang:spec_accept_length')
        hit = metrics.get('sglang:cache_hit_rate')
        ctx = info.get('context_length') or 0
        body.update(state='online', progress=100, phase='online', phase_label='在线',
                    running=int(metrics.get('sglang:num_running_reqs', 0)),
                    queue=int(metrics.get('sglang:num_queue_reqs', 0)),
                    tokens=int(used_tok), tps=tps, accept=accept, hit_rate=hit,
                    kv_pct=round(100.0 * used_tok / max_tok, 1) if max_tok else None,
                    options={})
        self._spark = (self._spark + [tps])[-self.spark_len:]
        body['spark'] = list(self._spark)

        detail = []
        if ctx:
            detail.append(f"上下文 {ctx // 1024}K · TP{info.get('tp_size')} · 并发上限 {info.get('max_running_requests')}")
        sched = []
        if info.get('schedule_policy'):
            sched.append(info['schedule_policy'])
        if info.get('prefill_decode_interval') is not None:
            sched.append(f"PDI {info['prefill_decode_interval']}")
        if info.get('chunked_prefill_size'):
            sched.append(f"chunk {info['chunked_prefill_size']}")
        if info.get('speculative_dspark_block_size'):
            sched.append(f"DSpark {info['speculative_dspark_block_size']}")
        if sched:
            detail.append(' · '.join(sched))
        body['detail'] = detail

        cache = _CACHE_LINE.findall(tail_text)
        if cache:
            hit_rate, experts, cpu_us, swaps, rounds, filled, total = cache[-1]
            body['xtier'] = {'hit': float(hit_rate), 'experts': float(experts), 'cpu_us': int(cpu_us),
                             'swaps': int(swaps), 'rounds': int(rounds), 'filled': int(filled), 'total': int(total)}
            body['detail'].append(f"XTier 热专家命中 {float(hit_rate) * 100:.1f}% · {float(experts):.0f} 专家/窗口 · "
                                  f"CPU {int(cpu_us)} µs/层 · 槽 {filled}/{total}")
        kb = metrics.get('sglang:kv_cache_memory_usage_gb')
        if kb:
            body['detail'].append(f"KV 池 {kb:.2f} GB · HiCache {int(metrics.get('sglang:hicache_host_used_tokens', 0)):,} token")
        body['detail'].append(f"累计 {int(metrics.get('sglang:num_requests_total', 0))} 请求 · "
                              f"prompt {int(metrics.get('sglang:prompt_tokens_total', 0)):,} · "
                              f"生成 {int(metrics.get('sglang:generation_tokens_total', 0)):,}")
        return body


class MultiReadonly:
    """Several model servers at once: one card per model, its parameters and its state.

    Read-only in the full sense: nothing here can start, stop or switch a server. Each card fills its middle
    column from status.args — the arguments the running server logged at start-up — the same as the original
    single-model read-only page.
    """

    def __init__(self, targets, models=None, make_targets=None, foot=None):
        """targets: [{'id', 'port', 'gpus', 'log', 'model'}] — 'model' is the card's metadata.

        make_targets (optional): rebuild the target list from scratch. The dashboard calls sync() so a server
        started after the board came up gets its card automatically; a server that went away loses its card."""
        self.targets = list(targets)
        self.views = {}
        for t in self.targets:
            self.views[t['id']] = Readonly(t['model'], log=t.get('log'), gpu_ids=t.get('gpus') or '0')
        self.make_targets = make_targets
        self.foot = foot
        self._sync_t = 0.0
        self._lock = threading.Lock()
        self._cache = {'t': 0.0, 'body': None}

    def sync(self, force=False):
        """Re-discover model servers, keeping the views (and spark history) of cards that stay. Returns True when
        the set of cards changed, so the page can reload itself."""
        if not self.make_targets:
            return False
        now = time.monotonic()
        if not force and now - self._sync_t < 5:
            return False
        self._sync_t = now
        fresh, seen = [], set()
        for t in self.make_targets():
            if t['id'] in seen:
                continue
            seen.add(t['id'])
            fresh.append(t)
        changed = set(self.views) != seen
        views = {}
        for t in fresh:
            v = self.views.get(t['id'])
            if v is None:
                v = Readonly(t['model'], log=t.get('log'), gpu_ids=t.get('gpus') or '0')
            else:
                if t.get('log') and t['log'] != v.log:
                    v.log, v._head = t['log'], None
                v.gpu_ids = t.get('gpus') or v.gpu_ids
            views[t['id']] = v
        self.targets, self.views = fresh, views
        if changed:
            with self._lock:
                self._cache = {'t': 0.0, 'body': None}
        return changed

    def console_data(self):
        models = {t['id']: {f: t['model'][f] for f in core.PUBLIC if f in t['model']} for t in self.targets}
        return {'models': models,
                'options': {t['id']: [] for t in self.targets},
                'defaults': {t['id']: {} for t in self.targets},
                'saved': {},
                'dashboard': True, 'readonly': True,
                'param_labels': _PARAM_LABELS,
                # no subtitle override: the card header keeps the console's own subtitle under XTIER
                'texts': {'foot': self.foot or '只读看板：每个模型一张原版卡片，中栏显示该服务启动时的实际参数；'
                                            '不含启停与切换。'}}

    def status(self):
        with self._lock:
            if self._cache['body'] is not None and time.monotonic() - self._cache['t'] < 1.2:
                return self._cache['body']
        body = {'models': {tid: view.status() for tid, view in self.views.items()}}
        with self._lock:
            self._cache.update(t=time.monotonic(), body=body)
        return body
