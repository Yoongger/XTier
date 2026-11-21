"""★ 服务器端 IOMMU A/B 基准（systemd oneshot 调用）。

★ 关键：decode 期间 PCIe 流量只有 ~278 KB/s（几乎没有 DMA），
  **IOMMU 在 decode 上无从影响**。IOMMU 只可能影响 **prefill**：
  专家权重从主机 RAM 上传到 GPU（GiB 级）+ 激活搬运 ⇒ 必须加 prefill 专项。

阶段：
  full        = decode 4 路 + decode 12 路 + 大上下文 prefill + PCIe 采样
  prefill     = 只跑大上下文 prefill + PCIe 采样（补基线缺口时用）

用法： bench_iommu_server.py <TAG> <OUT_JSON> [full|prefill]
"""
import json, os, re, subprocess, sys, time

PY = "/opt/miniforge3/envs/SGLang/bin/python"
BENCH = "/opt/Xtier/tools/bench_decode.py"
LOG = "/var/log/llm/DeepSeek-V4.1-Flash.log"
TAG = sys.argv[1] if len(sys.argv) > 1 else "IOMMU"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/opt/Xtier/snapshots/bench-iommu.json"
PHASE = sys.argv[3] if len(sys.argv) > 3 else "full"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, timeout=2600):
    p = subprocess.run(cmd, shell=True, executable="/bin/bash",
                       capture_output=True, text=True, errors="replace", timeout=timeout)
    return p.stdout


def wait_ready(max_wait=1800):
    log("等待 DeepSeek 服务就绪 ...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            p = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                                "--max-time", "5", "http://127.0.0.1:8001/health"],
                               capture_output=True, text=True, timeout=20)
            if p.stdout.strip() == "200":
                log(f"ready（耗时 {time.time()-t0:.0f}s）")
                time.sleep(10)
                return True
        except Exception:
            pass
        time.sleep(10)
    log("!! 服务未就绪")
    return False


def warmup():
    log("warmup ...")
    t0 = time.time()
    sh(f"cd /opt/Xtier && timeout 900 {PY} {BENCH} --sessions 4 --ctx 8000 --turns 1 --out 512 "
       f"> /tmp/warmup-abkit.log 2>&1", timeout=950)
    log(f"warmup done ({time.time()-t0:.0f}s, discarded)")


def measure(tag, rounds=3, sessions=4):
    results = []
    warmup()
    t_start = sh("date '+%Y-%m-%d %H:%M:%S'").strip()
    time.sleep(2)
    for r in range(rounds):
        out = sh(f"cd /opt/Xtier && timeout 2400 {PY} {BENCH} "
                 f"--sessions {sessions} --ctx 8000 --turns 2 --out 1024 2>&1 | tail -60", timeout=2500)
        wall = tok = n = p50 = p95 = mean = None
        gt200 = None
        ttfts = []
        for ln in out.splitlines():
            m = re.search(r"wall=([\d.]+)s", ln)
            if m: wall = float(m.group(1))
            m = re.search(r"n=(\d+)\s+p50=(\d+)ms\s+p95=(\d+)ms\s+max=(\d+)ms\s+mean=(\d+)ms", ln)
            if m:
                n = int(m.group(1)); p50 = int(m.group(2)); p95 = int(m.group(3)); mean = int(m.group(5))
            m = re.search(r">200ms:\s*(\d+)\s*\(([\d.]+)%\)", ln)
            if m: gt200 = float(m.group(2))
            m = re.search(r"聚合.*?(\d+) tok", ln)
            if m: tok = int(m.group(1))
            m = re.match(r"\s+s\d+\.\S+\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+", ln)
            if m:
                ttfts.append(float(m.group(1)))
        ttft = (sum(ttfts) / len(ttfts)) if ttfts else None
        client_tps = (tok / wall) if (tok and wall) else None
        results.append(dict(r=r + 1, wall=wall, tok=tok, n=n, p50=p50, p95=p95, mean=mean,
                            gt200=gt200, client_tps=client_tps, ttft=ttft))
        log(f"[{tag}] round{r+1}: wall={wall} tok={tok} n={n} ITL p50={p50} p95={p95} mean={mean} "
            f">200={gt200}% client_tps={client_tps and round(client_tps,1)} ttft={ttft}")
        time.sleep(5)
    t_end = sh("date '+%Y-%m-%d %H:%M:%S'").strip()

    timefilter = ("awk -v S='%s' -v E='%s' " % (t_start, t_end) +
                  "'{ts=substr($1,2)\" \"$2; if(ts>=S && ts<=E) print}'")
    srv = {}
    for want in (sessions, None):
        label = ("running-req=%s" % want) if want else "ALL"
        filt = (" | grep -a '#running-req: %s'" % want) if want else ""
        out = sh("grep -a 'Decode batch' " + LOG + " | " + timefilter + filt +
                 " | sed -E 's|.*#running-req: ([0-9]+).*accept len: ([0-9.]+)."
                 "*gen throughput \\(token/s\\): ([0-9.]+).*|\\1 \\2 \\3|' | "
                 "awk 'NF==3 {n++; r+=$1; a+=$2; t+=$3; v[n]=$3} "
                 "END { if(n) { asort(v); "
                 "printf \"n=%d avg_running=%.2f avg_accept=%.2f avg_tps=%.1f median_tps=%.1f\", "
                 "n, r/n, a/n, t/n, v[int(n/2)] } else print \"(no samples)\" }'")
        log(f"[{tag}] server[{label}]: {out.strip()}")
        srv[str(want)] = out.strip()

    cache = sh("grep -aE 'hit.?rate|elastic' " + LOG + " | " + timefilter +
               " | tail -4 | cut -c1-180")
    for ln in cache.splitlines():
        log(f"[{tag}] cache: {ln}")
    nrestart = sh("systemctl show DeepSeek-V4.1-Flash.service -p NRestarts --value").strip()
    log(f"[{tag}] NRestarts={nrestart}")
    return dict(tag=tag, t_start=t_start, t_end=t_end, rounds=results, srv=srv,
                nrestart=nrestart)


# ---------------- 大上下文 prefill 专项 ----------------

PRE = r'''
import json, os, re, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = 'http://127.0.0.1:8001/v1/chat/completions'
LOG = '/var/log/llm/DeepSeek-V4.1-Flash.log'

BASE = ("下面是一段用于压测的技术说明材料。内容涉及分布式系统的调度、缓存一致性、"
        "消息队列的背压处理、数据库事务隔离级别、以及大模型推理引擎的专家并行与流水并行设计。"
        "每一段都重复同样的主题，只是编号不同，目的是构造足够长的上下文以触发分块预填充。"
        "编号是 ")


def make_prompt(idx, target_chars=150000):
    # 每段约 60 字，带唯一编号 => 每个请求的 prompt 都不同（打散 prefix cache）
    parts = []
    i = 0
    total = 0
    while total < target_chars:
        s = BASE + str(idx) + "-" + str(i) + "。"
        parts.append(s)
        total += len(s)
        i += 1
    return "".join(parts)


def one(idx):
    body = json.dumps({
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': make_prompt(idx)}],
        'max_tokens': 32, 'temperature': 0.0, 'stream': False,
    }).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=2400) as r:
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


def run_prefill(concurrent=4, rounds=3, target_chars=150000):
    global samples
    out_all = []
    for rd in range(rounds):
        samples = []
        stop.clear()
        t = threading.Thread(target=sampler, daemon=True)
        t.start()
        time.sleep(2)
        t_start = subprocess.run(["date", "+%Y-%m-%d %H:%M:%S"], capture_output=True,
                                 text=True).stdout.strip()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrent) as ex:
            res = list(ex.map(one, [rd * 100 + i for i in range(concurrent)]))
        wall = time.time() - t0
        t_end = subprocess.run(["date", "+%Y-%m-%d %H:%M:%S"], capture_output=True,
                               text=True).stdout.strip()
        time.sleep(1)
        stop.set()
        time.sleep(2)

        # 服务端口径：这段窗口里真正 prefill 的 token 数（#new-token，不含 cached）
        awk = ("awk -v S='%s' -v E='%s' '{ts=substr($1,2)\" \"$2; if(ts>=S && ts<=E) print}'"
               % (t_start, t_end))
        p = subprocess.run("grep -a 'Prefill batch' " + LOG + " | " + awk +
                           " | sed -E 's/.*#new-seq: ([0-9]+), #new-token: ([0-9]+).*/\\1 \\2/'"
                           " | awk 'NF==2 {s+=$1; t+=$2} END {print s+0, t+0}'",
                           shell=True, capture_output=True, text=True)
        seq, newtok = (p.stdout.strip().split() + ["0", "0"])[:2]
        seq, newtok = int(seq), int(newtok)

        per_gpu = {}
        for gid, tx, rx in samples:
            per_gpu.setdefault(gid, []).append((tx, rx))
        pcie_tx = pcie_rx = 0.0
        for gid in (0, 2):
            if gid in per_gpu:
                arr = per_gpu[gid]
                pcie_tx += sum(a[0] for a in arr) / len(arr) / 1024.0
                pcie_rx += sum(a[1] for a in arr) / len(arr) / 1024.0
        # GPU0+GPU2 求和，单位 MB/s
        ttfts = [r[0] for r in res]
        out = dict(round=rd + 1, wall=round(wall, 2), seqs=seq, new_token=newtok,
                   prefill_tps=round(newtok / wall, 1) if wall else None,
                   ttft_s=round(sum(ttfts) / len(ttfts), 2),
                   max_ttft_s=round(max(ttfts), 2),
                   pcie_tx_MBps=round(pcie_tx, 2), pcie_rx_MBps=round(pcie_rx, 2))
        out_all.append(out)
        print(f"  [prefill] round{rd+1}: wall={out['wall']}s seqs={seq} new_token={newtok} "
              f"prefill_tps={out['prefill_tps']} TTFT均值={out['ttft_s']}s 最大={out['max_ttft_s']}s "
              f"PCIe tx={out['pcie_tx_MBps']}MB/s rx={out['pcie_rx_MBps']}MB/s", flush=True)
        time.sleep(3)
    return out_all


if __name__ == '__main__':
    res = run_prefill()
    with open('/tmp/_prefill_result.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('PREFILL_DONE')
'''


def run_prefill():
    log("--- 大上下文 prefill 专项（4 路 × ~150K 字符，唯一 prompt 打散 prefix cache）---")
    import tempfile
    path = "/tmp/_prefill_bench.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(PRE)
    out = sh(f"timeout 2400 {PY} {path} 2>&1", timeout=2500)
    for ln in out.splitlines():
        log("  " + ln)
    try:
        with open("/tmp/_prefill_result.json", encoding="utf-8") as f:
            rounds = json.load(f)
    except Exception:
        rounds = []
    sh("rm -f /tmp/_prefill_bench.py /tmp/_prefill_result.json")
    if not rounds:
        return {}
    def avg(k):
        vals = [r[k] for r in rounds if r.get(k) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    return {
        "rounds": rounds,
        "prefill_tps": avg("prefill_tps"),
        "ttft_s": avg("ttft_s"),
        "wall_s": avg("wall"),
        "pcie_tx_MBps": avg("pcie_tx_MBps"),
        "pcie_rx_MBps": avg("pcie_rx_MBps"),
    }


def main():
    log(f"=== IOMMU 基准开始 tag={TAG} phase={PHASE} ===")
    log("cmdline: " + sh("cat /proc/cmdline").strip())
    log("iommu: " + sh("echo -n 'groups='; ls /sys/kernel/iommu_groups/ 2>/dev/null | wc -l; "
                       "echo -n 'dmar='; ls -d /sys/class/iommu/* 2>/dev/null | wc -l").strip().replace("\n", " "))
    if not wait_ready():
        log("!! 服务未就绪，放弃基准")
        sys.exit(1)

    res = dict(tag=TAG, phase=PHASE, cmdline=sh("cat /proc/cmdline").strip(),
               t=time.strftime("%Y-%m-%d %H:%M:%S"))

    if PHASE in ("full", "decode"):
        for sess in (4, 12):
            log(f"--- decode sessions={sess} ---")
            res[f"s{sess}"] = measure(f"{TAG}_{sess}", rounds=3, sessions=sess)

    if PHASE in ("full", "prefill"):
        res["prefill"] = run_prefill()

    res["cache_tail"] = sh("grep -a 'XTier cache:' " + LOG + " | tail -2 | cut -c1-190")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"结果已写入 {OUT}")
    log("=== IOMMU 基准结束 ===")


if __name__ == "__main__":
    main()
