"""端到端验证（走用户真实路径：litellm 4000 → 8001）：
  A) 相同请求间隔 >60s（绕开 litellm 的 60s 响应缓存），带固定 seed → 是否可复现
  B) 同上但不传 seed → 引擎层漂移是否透传到用户
  C) 相同请求间隔 <60s → litellm 响应缓存是否给出字节相同的结果
"""
import json, urllib.request, hashlib, time
from collections import Counter

URL = 'http://127.0.0.1:4000/v1/chat/completions'
KEY = 'sk-Imation#2@26'
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
                                 {'Content-Type': 'application/json',
                                  'Authorization': f'Bearer {KEY}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d['choices'][0]['message']['content'] or ''


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def show(tag, o):
    print(f"   {tag}: sha={sha(o)} len={len(o)}  {o.strip()[:70]!r}")


if __name__ == '__main__':
    print("=== C) 间隔 <60s（应命中 litellm 的 60s 响应缓存）===")
    a1 = one(PROMPT, seed=777)
    show("t0", a1)
    a2 = one(PROMPT, seed=777)
    show("t+3s", a2)
    print(f"   => {'相同（缓存命中）' if a1 == a2 else '不同（缓存未命中）'}")

    print("\n=== A) 间隔 >60s + 固定 seed=777 ===")
    time.sleep(65)
    a3 = one(PROMPT, seed=777)
    show("t+68s", a3)
    print(f"   => {'与 t0 完全相同（seed 生效且可复现）' if a1 == a3 else '与 t0 不同'}")

    print("\n=== B) 间隔 >60s + 不传 seed ===")
    time.sleep(65)
    b1 = one(PROMPT)
    show("t+133s", b1)
    time.sleep(65)
    b2 = one(PROMPT)
    show("t+198s", b2)
    print(f"   => {'两次相同' if b1 == b2 else '两次不同（引擎层漂移透传到网关）'}")

    print("\n汇总：")
    print(f"  缓存内(<60s) 相同 : {a1 == a2}")
    print(f"  缓存外(>60s) 同 seed 相同 : {a1 == a3}")
    print(f"  缓存外(>60s) 无 seed 相同 : {b1 == b2}")
