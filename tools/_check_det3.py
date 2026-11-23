"""决定性测试：bs=1 下同 prompt 输出不一致的归因。

核心判据：**高度受限的 prompt**（模型 logit 极尖锐，argmax 与 runner-up 差距大）
如果输出仍然每次不同 ⇒ 引擎根本没在走 argmax（真 bug / 实际在采样）；
如果受限 prompt 输出完全一致，只有开放式 prompt 才分叉 ⇒ 是 logit 近平局的
正常敏感（batch/kernel 数值微差导致 greedy 翻转），属可接受范围。

另外测：temperature=0 之外，再确认 top_p/top_k 默认值没有偷偷把采样打开。
"""
import json, urllib.request, hashlib, time
from collections import Counter

URL = 'http://127.0.0.1:8001/v1/chat/completions'

CASES = [
    ("受限-数字", "只输出一个数字：1到10之间的任意一个数字，不要输出任何其他内容。"),
    ("受限-大写字母", "只输出三个大写英文字母，不要输出任何其他内容。"),
    ("受限-算术", "计算 17*23，只输出最终的数字结果，不要输出任何其他内容。"),
    ("受限-固定句", "原样输出这句话：今天天气很好。不要输出任何其他内容。"),
    ("开放-介绍", "用中文简要说明什么是向量数据库，并列举三个常见用途。"),
    ("开放-写作", "写一段关于秋天的短文，150 字左右。"),
]


def one(prompt, tokens=64, temperature=0.0, extra=None, timeout=1800):
    body = {
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': tokens, 'temperature': temperature, 'stream': False,
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d['choices'][0]['message']['content'] or ''


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def run_case(name, prompt, n=4, tokens=64, **kw):
    outs = []
    for _ in range(n):
        outs.append(one(prompt, tokens=tokens, **kw))
    h = [sha(o) for o in outs]
    g = Counter(h)
    uniq_text = len({o.strip() for o in outs})
    print(f"\n[{name}] 唯一 sha {len(g)}/{n}  唯一文本 {uniq_text}/{n}")
    for i, o in enumerate(outs):
        s = o.strip().replace("\n", "\\n")
        print(f"   run{i}: sha={h[i]} len={len(o)}  {s[:110]}")
    return len(g), outs


if __name__ == '__main__':
    print("=" * 72)
    print("第一部分：受限 prompt（logit 尖锐）vs 开放 prompt（logit 平坦）")
    print("=" * 72)
    results = {}
    for name, prompt in CASES:
        results[name] = run_case(name, prompt, n=4, tokens=64)[0]

    print("\n" + "=" * 72)
    print("第二部分：采样参数是否真的被关掉（显式 top_p/top_k/seed）")
    print("=" * 72)
    p = CASES[-1][1]
    run_case("显式top_p=1/top_k=-1/seed", p, n=3, tokens=64,
             extra={'top_p': 1.0, 'top_k': -1, 'seed': 1234})
    run_case("temperature=1（对照：应当变化很大）", p, n=3, tokens=64,
             temperature=1.0)

    print("\n" + "=" * 72)
    print("判定")
    print("=" * 72)
    sharp = [results[k] for k in results if k.startswith("受限")]
    loose = [results[k] for k in results if k.startswith("开放")]
    print(f"受限 prompt 唯一输出数: {sharp}（全 1 = 引擎确实走 argmax）")
    print(f"开放 prompt 唯一输出数: {loose}（>1 = logit 近平局敏感，属正常）")
    if all(v == 1 for v in sharp):
        print("=> 结论：引擎在走确定性 argmax；分叉只发生在 logit 接近的开放问题上。")
        print("   这是数值微差（batch/kernel 归约顺序）导致的 greedy 翻转，属可接受范围。")
    else:
        print("=> 结论：连 logit 尖锐的受限问题都不确定 ⇒ 需要进一步排查（真问题）。")
