#!/usr/bin/env bash
# ★ 生产配置自检
SNAP=/opt/Xtier/snapshots/prod
LIVE=/opt/Xtier/run-ds41-l20-tp2.sh
UNIT=/etc/systemd/system/DeepSeek-V4.1-Flash.service
fail=0

echo "== md5 比对 =="
a=$(md5sum "$SNAP/run-ds41-l20-tp2.sh.golden" | cut -d' ' -f1)
b=$(md5sum "$LIVE" | cut -d' ' -f1)
if [ "$a" = "$b" ]; then echo "  启动脚本  OK  $a"; else echo "  启动脚本  不一致！ golden=$a live=$b"; fail=1; fi
a=$(md5sum "$SNAP/DeepSeek-V4.1-Flash.service.golden" | cut -d' ' -f1)
b=$(md5sum "$UNIT" | cut -d' ' -f1)
if [ "$a" = "$b" ]; then echo "  unit      OK  $a"; else echo "  unit      不一致！"; fail=1; fi

echo "== 关键参数 =="
grep -oE 'XTIER_CACHE_ELASTIC_MAX_SLOTS=[^ ]*|--schedule-policy [^ ]*|--max-running-requests [^ ]*|--prefill-decode-interval [^ ]*|--chunked-prefill-size [^ ]*|--cuda-graph-max-bs-decode [^ ]*|XTIER_CACHE_SLOTS=[^ ]*|XTIER_THREADS=[^ ]*' "$LIVE" | sort -u | sed 's/^/  /'

echo "== 服务状态 =="
systemctl show DeepSeek-V4.1-Flash.service -p NRestarts -p ActiveState -p SubState --value | tr '\n' ' '; echo
echo -n "  health: "; curl -s -o /dev/null -w '%{http_code}\n' --max-time 5 http://127.0.0.1:8001/health
echo -n "  litellm: "; curl -s -o /dev/null -w '%{http_code}\n' --max-time 5 -H 'Authorization: Bearer sk-Imation#2@26' http://127.0.0.1:4000/health/liveliness
echo -n "  enabled: "; systemctl is-enabled DeepSeek-V4.1-Flash.service
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/  /'
exit $fail
