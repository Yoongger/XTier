"""服务器端 PCIe 探针（由 bench_iommu_server.py 调用）。"""
import json, os, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = 'http://127.0.0.1:8001/v1/chat/completions'
PROMPT = "用中文写一段 800 字的说明，介绍如何部署一套多卡大模型推理服务。"


def one(_):
    body = json.dumps({
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': PROMPT}],
        'max_tokens': 1024, 'temperature': 0.0, 'stream': False,
    }).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read().decode())
    return time.time() - t0, len(d['choices'][0]['message']['content'] or '')


samples = []
stop = threading.Event()


def sampler():
    p = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'p', '-d', '1'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in p.stdout:
            if stop.is_set():
                break
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    samples.append((int(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
    finally:
        p.kill()


t = threading.Thread(target=sampler, daemon=True)
t.start()
time.sleep(3)

lat = []
done = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(one, i) for i in range(4)]
    while time.time() - t0 < 75:
        time.sleep(2)
        if all(f.done() for f in futs):
            futs = [ex.submit(one, i) for i in range(4)]
    for f in futs:
        if f.done():
            try:
                dt, n = f.result()
                lat.append((dt, n))
                done += 1
            except Exception:
                pass
time.sleep(2)
stop.set()
time.sleep(2)

per_gpu = {}
for gid, tx, rx in samples:
    per_gpu.setdefault(gid, []).append((tx, rx))
print("=== PCIe 采样（nvidia-smi dmon -s p，单位 KB/s）===")
for gid in sorted(per_gpu):
    arr = per_gpu[gid]
    tx = sum(a[0] for a in arr) / len(arr)
    rx = sum(a[1] for a in arr) / len(arr)
    print(f"  GPU{gid}: tx={tx:10.0f} KB/s ({tx/1024:7.1f} MB/s)   rx={rx:10.0f} KB/s ({rx/1024:7.1f} MB/s)   n={len(arr)}")
if lat:
    print(f"=== 负载 ===")
    print(f"  完成 {done} 个请求，平均耗时 {sum(a[0] for a in lat)/len(lat):.1f}s，"
          f"输出 {sum(a[1] for a in lat)} 字")
