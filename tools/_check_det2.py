"""决定性对照：区分「连续批处理(batch composition)非确定性」与「bs=1 下的真非确定性」。

A) 顺序 4 次同 prompt（每次都独占，bs=1）→ 若 4 次完全相同，说明 bs=1 是确定的
B) 并发 4 次同 prompt（同一 batch）→ 与 A 对比，差异来源只能是 batch composition
C) 顺序 4 次 + 每���换一个「前缀不同但语义等价」的 prompt → 排除缓存状态因素
"""
import json, urllib.request, hashlib, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

URL = 'http://127.0.0.1:8001/v1/chat/completions'
PROMPT = "用中文简要说明什么是向量数据库，并列举三个常见用途。"


def one(prompt, tokens=128, timeout=1800):
    body = json.dumps({
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': tokens, 'temperature': 0.0, 'stream': False,
    }).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d['choices'][0]['message']['content'] or ''


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def report(tag, outs):
    h = [sha(o) for o in outs]
    g = Counter(h)
    print(f"  [{tag}] 唯一输出数 {len(g)}/{len(outs)}  分组 {sorted(g.values(), reverse=True)}")
    print(f"     sha: {h}")
    return g


if __name__ == '__main__':
    print("=== A) 顺序 4 次（bs=1 独占）===")
    seq = []
    for i in range(4):
        t0 = time.time()
        o = one(PROMPT)
        seq.append(o)
        print(f"   run{i}: {len(o)} chars sha={sha(o)}  {time.time()-t0:.1f}s")
    ga = report("顺序x4", seq)

    print("\n=== B) 并发 4 次（同一批）===")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        par = list(ex.map(lambda _: one(PROMPT), range(4)))
    gb = report("并发x4", par)
    print(f"   ({time.time()-t0:.1f}s)")

    print("\n=== C) 顺序 4 次，带唯一后缀（排除 prefix-cache 命中差异）===")
    seq2 = []
    for i in range(4):
        o = one(PROMPT + f"（第{i}次提问）")
        seq2.append(o)
        print(f"   run{i}: {len(o)} chars sha={sha(o)}")
    gc = report("顺序x4(带后缀)", seq2)

    print("\n=== 结论 ===")
    if len(ga) == 1:
        print("  A) bs=1 顺序请求输出完全相同 => 引擎在固定 batch 下是确定的；")
        print("     并发下的差异纯粹来自 batch composition（连续批处理的固有性质）。")
    else:
        print("  A) bs=1 顺序请求输出也不同 => 存在 bs=1 下的非确定性，需要进一步定位")
        print("     （可能来源：DSpark 投机 verify 的接受判定 / prefix-cache 状态差异）。")
    if len(gc) == 1:
        print("  C) 带唯一后缀后仍然确定 => 与 prefix-cache 无关")
    else:
        print("  C) 带唯一后缀后仍不确定 => 与 prompt 文本/缓存无关，是引擎侧数值差异")
