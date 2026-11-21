#!/bin/bash
# 拉起 XTier 推理服务（8001）。
#
# 控制台（8100）是独立服务 xtier-console.service，生命周期与本服务解耦：它先于本服务启动，
# 本服务 down 掉后仍存活，读本服务日志并通过 systemctl 启停。这里只启动模型。
set -e
cd /opt/Xtier

# 主进程：模型服务前台运行；run-ds41-l20-tp2.sh 内部再 exec 到 sglang.launch_server。
exec /bin/bash /opt/Xtier/run-ds41-l20-tp2.sh
