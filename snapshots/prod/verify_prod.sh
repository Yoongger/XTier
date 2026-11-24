#!/usr/bin/env bash
# ★ 生产配置自检 + ★ unit / 启动脚本 / 运行时 三方对齐校验
#
# 用法： sudo bash /opt/Xtier/snapshots/prod/verify_prod.sh
#       bash /opt/Xtier/snapshots/prod/verify_prod.sh   （environ 部分需要 root）
SNAP=/opt/Xtier/snapshots/prod
LIVE=/opt/Xtier/run-ds41-l20-tp2.sh
UNIT=/etc/systemd/system/DeepSeek-V4.1-Flash.service
fail=0

echo "=============================================================="
echo "== 1) md5 比对（live vs golden） =="
echo "=============================================================="
a=$(md5sum "$SNAP/run-ds41-l20-tp2.sh.golden" 2>/dev/null | cut -d' ' -f1)
b=$(md5sum "$LIVE" | cut -d' ' -f1)
if [ "$a" = "$b" ]; then echo "  启动脚本  OK  $a"; else echo "  启动脚本  不一致！ golden=$a live=$b"; fail=1; fi
a=$(md5sum "$SNAP/DeepSeek-V4.1-Flash.service.golden" 2>/dev/null | cut -d' ' -f1)
b=$(md5sum "$UNIT" | cut -d' ' -f1)
if [ "$a" = "$b" ]; then echo "  unit      OK  $a"; else echo "  unit      不一致！ golden=$a live=$b"; fail=1; fi

echo
echo "=============================================================="
echo "== 2) unit <-> 启动脚本 交叉校验 =="
echo "   （unit 里每个变量都必须在脚本里有同名 \${VAR:-默认}，且默认值相同）"
echo "=============================================================="
grep -E '^Environment=' "$UNIT" | sed 's/^Environment=//' | tr -d '"' | grep -v '^$' | sort > /tmp/_unit_env.txt
# 脚本里所有「可覆盖」的变量（含 ${VAR:-} 这种空默认写法）
grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*:-[^}]*\}' "$LIVE" \
  | sed 's/^\${//; s/}$//; s/:-/=/; s/="/=/; s/"$//' | sort -u > /tmp/_script_env_all.txt
# 其中「有非空默认值」的（用于比对取值）
grep -v '=$' /tmp/_script_env_all.txt > /tmp/_script_env.txt
nunit=0; nok=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  nunit=$((nunit+1))
  k=${line%%=*}
  v=${line#*=}
  if ! grep -q "^${k}=" /tmp/_script_env_all.txt; then
    echo "  FAIL  $k = $v   -> 脚本里没有同名可覆盖变量（unit 设了会被脚本忽略）"; fail=1
    continue
  fi
  sv=$(grep -m1 "^${k}=" /tmp/_script_env.txt | cut -d= -f2-)
  if [ -z "$sv" ]; then
    nok=$((nok+1)); echo "  OK    $k = $v   (脚本可覆盖但无默认值，采用 unit 的值)"
  elif [ "$sv" != "$v" ]; then
    echo "  FAIL  $k : unit=$v  脚本默认=$sv   -> 两边不一致"; fail=1
  else
    nok=$((nok+1)); echo "  OK    $k = $v"
  fi
done < /tmp/_unit_env.txt
echo "  ---- unit 共 $nunit 个变量，$nok 个与脚本一致 ----"
echo "  （脚本里可覆盖但 unit 未声明的变量：只是没暴露在审计面，不算错）"
comm -13 /tmp/_unit_env.txt /tmp/_script_env_all.txt | sed 's/^/     /'

echo
echo "=============================================================="
echo "== 3) 运行中进程命令行（最终生效值，ground truth） =="
echo "=============================================================="
P=$(systemctl show DeepSeek-V4.1-Flash.service -p MainPID --value 2>/dev/null)
if [ -z "$P" ] || [ "$P" = "0" ]; then
  P=$(pgrep -f 'sglang.launch_server' | head -1)
fi
if [ -n "$P" ]; then
  CMD=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
  echo "  pid=$P"
  for expect in "--tp-size 2" "--port 8001" "--schedule-policy fcfs" \
                "--max-running-requests 8" "--chunked-prefill-size 16384" \
                "--max-prefill-tokens 16384" "--prefill-decode-interval 32" \
                "--prefill-max-requests 8" "--cuda-graph-max-bs-decode 8" \
                "--context-length 262144" "--max-total-tokens 1572864" \
                "--mem-fraction-static 0.88" "--max-mamba-cache-size 16" \
                "--speculative-algorithm DSPARK" "--speculative-num-draft-tokens 6" \
                "--speculative-dspark-sps-table-path /opt/Xtier/sps-table-l20-gpu02.json"; do
    case " $CMD " in
      *" $expect "*) echo "  OK    $expect" ;;
      *) echo "  FAIL  缺 $expect"; fail=1 ;;
    esac
  done
  echo "  ---- 环境变量（/proc/$P/environ）----"
  ENVP=$(tr '\0' '\n' < /proc/$P/environ 2>/dev/null)
  if [ -z "$ENVP" ]; then
    echo "     (需要 root 才能读 root 进程的 environ；用 sudo 重跑本脚本可见)"
  else
    for k in XTIER_CACHE_ELASTIC_MAX_SLOTS XTIER_CACHE_ELASTIC_BASE XTIER_CACHE_SLOTS \
             XTIER_THREADS XTIER_TP_NODES XTIER_TP_NODES SGLANG_RAGGED_VERIFY_MODE \
             DSPARK SPS_TABLE KV_POOL MAMBA_CACHE NCCL_P2P_LEVEL XTIER_THP \
             CUDA_VISIBLE_DEVICES; do
      v=$(echo "$ENVP" | grep -m1 "^${k}=" | cut -d= -f2-)
      echo "     $k=$v"
    done
  fi
else
  echo "  FAIL  没找到 sglang.launch_server 进程"; fail=1
fi

echo
echo "=============================================================="
echo "== 4) 服务状态 =="
echo "=============================================================="
systemctl show DeepSeek-V4.1-Flash.service -p NRestarts -p ActiveState -p SubState --value | tr '\n' ' '; echo
echo -n "  health: "; curl -s -o /dev/null -w '%{http_code}\n' --max-time 5 http://127.0.0.1:8001/health
echo -n "  litellm: "; curl -s -o /dev/null -w '%{http_code}\n' --max-time 5 -H 'Authorization: Bearer sk-Imation#2@26' http://127.0.0.1:4000/health/liveliness
echo -n "  enabled: "; systemctl is-enabled DeepSeek-V4.1-Flash.service

echo
echo "=============================================================="
echo "== 5) GPU =="
echo "=============================================================="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/  /'

echo
echo "=============================================================="
echo "== 6) IOMMU / 内核参数 =="
echo "=============================================================="
echo "  cmdline: $(cat /proc/cmdline)"
echo "  grub:    $(cat /etc/default/grub.d/99-nvidia.cfg 2>/dev/null)"
echo "  groups:  $(ls /sys/kernel/iommu_groups/ 2>/dev/null | wc -l)  dmar: $(ls -d /sys/class/iommu/* 2>/dev/null | wc -l)"

echo
if [ "$fail" = "0" ]; then
  echo "★ 自检全部通过（fail=0）"
else
  echo "!! 自检有失败项（fail=1）"
fi
exit $fail
