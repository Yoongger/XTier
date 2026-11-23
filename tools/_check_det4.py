"""确认「传 seed ⇒ 输出完全可复现」不是偶然：8 次带固定 seed vs 8 次不带 seed。"""
import json, urllib.request, hashlib
from collections import Counter

URL = 'http://127.0.0.1:8001/v1/chat/completions'
PROMPT = "写一段关于秋天的短文，150 字左右。"


def one(prompt, tokens=96, seed=None, timeout=1800):
    body = {
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': tokens, 'temperature': 0.0, 'stream': False,
    }
    if seed is not None:
        body['seed'] = seed
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d['choices'][0]['message']['content'] or ''


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def run(tag, seed, n=8):
    outs = [one(PROMPT, seed=seed) for _ in range(n)]
    h = [sha(o) for o in outs]
    g = Counter(h)
    print(f"[{tag}] 唯一 sha {len(g)}/{n}  分组 {sorted(g.values(), reverse=True)}")
    print(f"   sha: {h}")
    return len(g)


if __name__ == '__main__':
    print("=== 8 次，不传 seed ===")
    a = run("无seed", None, 8)
    print("\n=== 8 次，固定 seed=424242 ===")
    b = run("seed=424242", 424242, 8)
    print("\n=== 8 次，另一个固定 seed=999 ===")
    c = run("seed=999", 999, 8)
    print("\n判定：")
    print(f"  无 seed 唯一数 = {a}（>1 说明不传 seed 不可复现）")
    print(f"  固定 seed 唯一数 = {b}（=1 说明带 seed 逐 token 可复现）")
    print(f"  另一 seed 唯一数 = {c}")
