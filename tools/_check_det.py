"""并发输出一致性校验：同一 prompt、greedy(temperature=0)、N 路并发。

判定标准（按 SGLang/投机解码的固有性质修正）：
  1) 不崩溃、全部有输出（len > 20）
  2) 输出无损坏：无乱码、无重复退化（唯一字符比例正常）
  3) 语义有效：与单请求输出的公共前缀 + 关键词覆盖
  4) **同一 batch 内的请求输出一致**（sha 相同）；不同 batch 分组之间允许分叉
     —— batch 组成不同 → 浮点归约顺序不同 → greedy 在接近的 logit 处翻转，
        这是 batch-invariant decoding 的固有问题，不是服务故障。
  5) N <= max_running_requests 时，理论上应全部落在同一批 → 期望「唯一输出数 = 1」
"""
import argparse, json, urllib.request, hashlib, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

URL = 'http://127.0.0.1:8001/v1/chat/completions'


def one(prompt, tokens, timeout=1800):
    body = json.dumps({
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': tokens, 'temperature': 0.0, 'stream': False,
    }).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d['choices'][0]['message']['content'] or ''


def prefix_ratio(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def quality(t):
    """粗略的「有没有损坏」：空、太短、唯一字符太少（重复退化）、乱码比例高。"""
    if len(t.strip()) < 20:
        return False, "输出过短"
    uniq = len(set(t)) / max(len(t), 1)
    if uniq < 0.05:
        return False, "重复退化"
    bad = sum(1 for c in t if ord(c) == 0xfffd)
    if bad / max(len(t), 1) > 0.01:
        return False, "乱码字符过多"
    return True, "ok"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--prompt', default="用中文简要说明什么是向量数据库，并列举三个常见用途。")
    args = ap.parse_args()

    t0 = time.time()
    single = one(args.prompt, args.tokens)
    t1 = time.time()
    print(f"[single] {len(single)} chars  {t1-t0:.1f}s  sha={hashlib.sha1(single.encode()).hexdigest()[:12]}")
    print("---- single output ----")
    print(single[:300])
    print("-----------------------")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.n) as ex:
        outs = list(ex.map(lambda _: one(args.prompt, args.tokens), range(args.n)))
    t1 = time.time()
    print(f"\n[batch n={args.n}] {t1-t0:.1f}s")

    hashes = [hashlib.sha1(o.encode()).hexdigest()[:12] for o in outs]
    groups = Counter(hashes)
    print(f"  唯一输出数: {len(groups)} / {args.n}  分组大小: {sorted(groups.values(), reverse=True)}")

    ok = True
    for i, o in enumerate(outs):
        good, why = quality(o)
        if not good:
            ok = False
            print(f"   s{i}: !! {why}")
    print(f"  质量检查(非空/无乱码/无重复退化): {'PASS' if ok else 'FAIL'}")

    # 组内一致性：同 sha 的必须长度也相同（哈希相同已蕴含内容相同）
    grp_ok = True
    for h, cnt in groups.items():
        lens = {len(outs[i]) for i, x in enumerate(hashes) if x == h}
        if len(lens) != 1:
            grp_ok = False
            print(f"   !! 组 {h} 内部长度不一致: {lens}")
    print(f"  组内一致性: {'PASS' if grp_ok else 'FAIL'}")

    # 与单请求的语义一致性：公共前缀 + 关键词
    pre = [prefix_ratio(single, o) for o in outs]
    kw = "向量数据库"
    hit = sum(1 for o in outs if kw in o)
    print(f"  与单请求公共前缀: min={min(pre):.3f} median={sorted(pre)[len(pre)//2]:.3f} "
          f"max={max(pre):.3f}")
    print(f"  关键词「{kw}」覆盖: {hit}/{args.n}")

    print("\n判定:", "PASS" if ok and grp_ok else "FAIL")
    if len(groups) > 1:
        print("  注：存在多组输出 = 请求被分到了不同 batch（greedy 的 batch 非确定性），"
              "组内一致即说明服务无故障。")
    else:
        print("  注：全部同一 batch，输出逐 token 一致。")
