#!/usr/bin/env bash
# ★ 恢复 DeepSeek-V4.1-Flash 生产配置（2026-11-20 固化版）
# 用法： sudo bash /opt/Xtier/snapshots/prod/restore_prod.sh
set -e
SNAP=/opt/Xtier/snapshots/prod
LIVE=/opt/Xtier/run-ds41-l20-tp2.sh
UNIT=/etc/systemd/system/DeepSeek-V4.1-Flash.service

echo "[1/5] 恢复启动脚本"
cp -a "$SNAP/run-ds41-l20-tp2.sh.golden" "$LIVE"
echo "[2/5] 恢复 systemd unit"
cp -a "$SNAP/DeepSeek-V4.1-Flash.service.golden" "$UNIT"
systemctl daemon-reload
echo "[3/5] 确保开机自启"
systemctl enable DeepSeek-V4.1-Flash.service
echo "[4/5] 重启服务"
systemctl restart DeepSeek-V4.1-Flash.service
echo "[5/5] 等待就绪（最多 20 分钟）"
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8001/health || true)
  if [ "$code" = "200" ]; then
    echo "ready after $((i*10))s"
    systemctl show DeepSeek-V4.1-Flash.service -p NRestarts -p ActiveState --value | tr '\n' ' '; echo
    exit 0
  fi
  sleep 10
done
echo "!! 服务未就绪"
exit 1
