#!/bin/bash
# 拉起 XTier 推理服务（8001）。
#
# 状态看板（8100）已独立成 xtier-dashboard.service：它自动发现本机所有模型服务并同时显示，
# 与模型服务解耦（模型重启不影响看板，看板也不跟服务抢端口）。这里不再启动控制台。
set -e
cd /opt/Xtier

# 主进程：模型服务前台运行；run-ds41-l20-tp2.sh 内部再 exec 到 sglang.launch_server。
exec /bin/bash /opt/Xtier/run-ds41-l20-tp2.sh
