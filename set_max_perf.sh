#!/bin/bash
set -euo pipefail

DEVFREQ=/sys/class/devfreq
BIG_CORE_MIN_KHZ=2000000

find_dev() {
    _want=$1
    for _d in "$DEVFREQ"/*/; do
        [ -r "$_d/name" ] || continue
        if [ "$(cat "$_d/name" 2>/dev/null)" = "$_want" ]; then
            printf '%s\n' "${_d%/}"
            return 0
        fi
    done
    return 1
}

set_governor() {
    _dev=$1
    _want=$2
    _avail=$(cat "$_dev/available_governors" 2>/dev/null)
    case " $_avail " in
        *" $_want "*) ;;
        *) return 1 ;;
    esac
    echo "$_want" > "$_dev/governor" 2>/dev/null || return 1
    [ "$(cat "$_dev/governor" 2>/dev/null)" = "$_want" ] || return 1
    return 0
}

# dmc 在部分 RK3588 内核/DTB 里被禁用，找不到时降级跳过，不报错
DMC=$(find_dev dmc) || DMC=""
NPU=$(find_dev fdab0000.npu) || NPU=""

fail=0

if [ -z "$DMC" ]; then
    echo "[npu-perf] 未找到 dmc devfreq（设备树已禁用 DDR 调频），跳过锁频" >&2
elif ! set_governor "$DMC" performance; then
    echo "[npu-perf] dmc 无法设为 performance，当前=$(cat "$DMC/governor" 2>/dev/null)" >&2
    fail=1
else
    echo "[npu-perf] dmc governor = performance"
fi

if [ -n "$NPU" ]; then
    if ! set_governor "$NPU" performance; then
        echo "[npu-perf] npu 无法设为 performance，当前=$(cat "$NPU/governor" 2>/dev/null)" >&2
        fail=1
    else
        echo "[npu-perf] npu governor = performance"
    fi
fi

_n=0
for _c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
    [ -d "$_c" ] || continue
    _max=$(cat "$_c/cpuinfo_max_freq" 2>/dev/null) || continue
    [ "$_max" -ge "$BIG_CORE_MIN_KHZ" ] || continue
    [ "$(cat "$_c/scaling_governor" 2>/dev/null)" = performance ] && continue
    if echo performance > "$_c/scaling_governor" 2>/dev/null; then
        _n=$((_n + 1))
    fi
done
echo "[npu-perf] 大核 CPU 锁定 performance 数量: $_n"

echo "[npu-perf] 当前状态："
if [ -n "$DMC" ]; then
    printf '  dmc  governor=%s freq=%s / max %s\n' \
        "$(cat "$DMC/governor" 2>/dev/null)" \
        "$(cat "$DMC/cur_freq" 2>/dev/null)" \
        "$(cat "$DMC/max_freq" 2>/dev/null)"
fi
if [ -n "$NPU" ]; then
    printf '  npu  governor=%s freq=%s / max %s  load=%s\n' \
        "$(cat "$NPU/governor" 2>/dev/null)" \
        "$(cat "$NPU/cur_freq" 2>/dev/null)" \
        "$(cat "$NPU/max_freq" 2>/dev/null)" \
        "$(cat "$NPU/load" 2>/dev/null)"
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

if [ -n "$DMC" ]; then
    echo '[npu-perf] 已完成：DMC/NPU/大核 CPU 均锁定最高性能'
else
    echo '[npu-perf] 已完成：NPU/大核 CPU 锁定最高性能（DMC 未启用，已跳过）'
fi