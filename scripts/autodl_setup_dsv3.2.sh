#!/usr/bin/env bash
###############################################################################
# AutoDL 一键部署 DeepSeek-V3.2 + KTransformers 环境
#
# 用法:
#   bash autodl_setup_dsv3.2.sh          # 全部执行（环境+下载）
#   bash autodl_setup_dsv3.2.sh env      # 仅安装环境
#   bash autodl_setup_dsv3.2.sh download # 仅下载模型
#   bash autodl_setup_dsv3.2.sh status   # 查看下载进度
#   bash autodl_setup_dsv3.2.sh launch   # 启动 KTransformers
#
# 特性:
#   - aria2c 多线程并发下载（~17MB/s vs 1.4MB/s）
#   - 断点续传，中断后重跑自动跳过已完成文件
#   - safetensors + GGUF 同时并发下载
#   - 自动检测磁盘空间
###############################################################################
set -euo pipefail

# ========================= 配置区 =========================
MODEL_NAME="DeepSeek-V3.2"
GGUF_QUANT="Q4_K_M"

# 存储路径（AutoDL 数据盘）
BASE_DIR="/root/autodl-tmp/models"
ST_DIR="${BASE_DIR}/${MODEL_NAME}"
GGUF_DIR="${BASE_DIR}/${MODEL_NAME}-GGUF/${GGUF_QUANT}"
LOG_DIR="/root/autodl-tmp/logs"

# ModelScope 仓库
MS_ST_REPO="deepseek-ai/DeepSeek-V3.2"
MS_GGUF_REPO="QuantFactory/DeepSeek-V3.2-GGUF"

# aria2c 参数
ARIA2_JOBS=8          # 同时下载文件数
ARIA2_CONN=16         # 每个文件连接数
ARIA2_RETRY=10        # 重试次数
ARIA2_TIMEOUT=60      # 超时秒数
ARIA2_MIN_SIZE="1G"   # 跳过大于此大小的已有文件（视为已完成）

# KTransformers conda 环境
CONDA_ENV="kt"
KT_VERSION="0.5.2.post2"

# KTransformers 启动参数
KT_HOST="0.0.0.0"
KT_PORT=30000
KT_CPU_THREADS=100    # CPU推理线程数，根据核数调整
KT_GPU_EXPERTS=0      # GPU专家数，0=自动

# ========================= 颜色 =========================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }
info() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }

# ========================= 工具函数 =========================
check_disk() {
    local avail_gb
    avail_gb=$(df --output=avail /root/autodl-tmp 2>/dev/null | tail -1 | awk '{printf "%.0f", $1/1024/1024}')
    local need_gb=1100  # safetensors(690) + GGUF(378) + buffer
    log "磁盘空间: 可用 ${avail_gb}GB, 需要 ~${need_gb}GB"
    if (( avail_gb < need_gb )); then
        warn "磁盘空间可能不足！可用: ${avail_gb}GB < 需要: ${need_gb}GB"
        warn "如果部分文件已下载，可以忽略此警告"
    fi
}

install_aria2() {
    if command -v aria2c &>/dev/null; then
        log "aria2c 已安装: $(aria2c --version | head -1)"
        return
    fi
    log "安装 aria2c..."
    apt-get update -qq && apt-get install -y -qq aria2
    log "aria2c 安装完成"
}

# ========================= 环境安装 =========================
setup_env() {
    log "========== 安装环境 =========="
    install_aria2

    # conda 环境
    if conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
        log "conda 环境 '${CONDA_ENV}' 已存在"
    else
        log "创建 conda 环境 '${CONDA_ENV}'..."
        conda create -n "${CONDA_ENV}" python=3.11 -y
    fi

    # 激活环境并安装 ktransformers
    log "安装 KTransformers..."
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"

    if python -c "import ktransformers; print(ktransformers.__version__)" 2>/dev/null; then
        log "KTransformers 已安装"
    else
        pip install ktransformers=="${KT_VERSION}" 2>/dev/null \
            || pip install ktransformers \
            || warn "KTransformers pip 安装失败，可能需要从源码编译"
    fi

    log "========== 环境就绪 =========="
}

# ========================= 生成下载列表 =========================
generate_st_filelist() {
    local list_file="${LOG_DIR}/st_download.list"
    mkdir -p "${ST_DIR}" "${LOG_DIR}"

    log "生成 safetensors 下载列表..."
    > "${list_file}"

    # 163 个 safetensors 分片
    for i in $(seq -w 1 163); do
        local fname="model-${i}-of-000163.safetensors"
        local fpath="${ST_DIR}/${fname}"

        # 跳过已完成的文件（>1GB）
        if [[ -f "${fpath}" ]] && (( $(stat -c%s "${fpath}" 2>/dev/null || echo 0) > 1073741824 )); then
            continue
        fi

        local url="https://modelscope.cn/models/${MS_ST_REPO}/resolve/master/${fname}"
        echo "${url}" >> "${list_file}"
        echo "  dir=${ST_DIR}" >> "${list_file}"
        echo "  out=${fname}" >> "${list_file}"
    done

    # 附加配置文件
    for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
             generation_config.json model.safetensors.index.json; do
        if [[ ! -f "${ST_DIR}/${f}" ]]; then
            local url="https://modelscope.cn/models/${MS_ST_REPO}/resolve/master/${f}"
            echo "${url}" >> "${list_file}"
            echo "  dir=${ST_DIR}" >> "${list_file}"
            echo "  out=${f}" >> "${list_file}"
        fi
    done

    local count
    count=$(grep -c '^https://' "${list_file}" 2>/dev/null || echo 0)
    log "safetensors 待下载: ${count} 个文件"
    echo "${list_file}"
}

generate_gguf_filelist() {
    local list_file="${LOG_DIR}/gguf_download.list"
    mkdir -p "${GGUF_DIR}" "${LOG_DIR}"

    log "生成 GGUF 下载列表..."
    > "${list_file}"

    # Q4_K_M 9 个分片
    for i in $(seq -w 1 9); do
        local fname="DeepSeek-V3.2-${GGUF_QUANT}-0000${i}-of-00009.gguf"
        local fpath="${GGUF_DIR}/${fname}"

        if [[ -f "${fpath}" ]] && (( $(stat -c%s "${fpath}" 2>/dev/null || echo 0) > 1073741824 )); then
            continue
        fi

        local url="https://modelscope.cn/models/${MS_GGUF_REPO}/resolve/master/${fname}"
        echo "${url}" >> "${list_file}"
        echo "  dir=${GGUF_DIR}" >> "${list_file}"
        echo "  out=${fname}" >> "${list_file}"
    done

    local count
    count=$(grep -c '^https://' "${list_file}" 2>/dev/null || echo 0)
    log "GGUF 待下载: ${count} 个文件"
    echo "${list_file}"
}

# ========================= 下载执行 =========================
run_aria2() {
    local list_file="$1"
    local log_file="$2"
    local label="$3"

    local count
    count=$(grep -c '^https://' "${list_file}" 2>/dev/null || echo 0)
    if (( count == 0 )); then
        log "${label}: 所有文件已下载完成，跳过"
        return 0
    fi

    log "${label}: 开始下载 ${count} 个文件 → 日志: ${log_file}"
    aria2c \
        --input-file="${list_file}" \
        --max-concurrent-downloads="${ARIA2_JOBS}" \
        --max-connection-per-server="${ARIA2_CONN}" \
        --split="${ARIA2_CONN}" \
        --min-split-size=10M \
        --max-tries="${ARIA2_RETRY}" \
        --retry-wait=5 \
        --timeout="${ARIA2_TIMEOUT}" \
        --connect-timeout=30 \
        --continue=true \
        --auto-file-renaming=false \
        --allow-overwrite=false \
        --file-allocation=none \
        --console-log-level=warn \
        --summary-interval=30 \
        --log="${log_file}" \
        --log-level=notice \
        2>&1 | tail -f &

    echo $!
}

download_all() {
    log "========== 开始下载模型 =========="
    check_disk
    install_aria2

    local st_list gguf_list
    st_list=$(generate_st_filelist)
    gguf_list=$(generate_gguf_filelist)

    local st_count gguf_count
    st_count=$(grep -c '^https://' "${st_list}" 2>/dev/null || echo 0)
    gguf_count=$(grep -c '^https://' "${gguf_list}" 2>/dev/null || echo 0)

    # 并发启动两个 aria2c（safetensors + GGUF 同时下载）
    local pids=()

    if (( st_count > 0 )); then
        log "启动 safetensors 下载 (${st_count} 文件)..."
        aria2c \
            --input-file="${st_list}" \
            --max-concurrent-downloads="${ARIA2_JOBS}" \
            --max-connection-per-server="${ARIA2_CONN}" \
            --split="${ARIA2_CONN}" \
            --min-split-size=10M \
            --max-tries="${ARIA2_RETRY}" \
            --retry-wait=5 \
            --timeout="${ARIA2_TIMEOUT}" \
            --connect-timeout=30 \
            --continue=true \
            --auto-file-renaming=false \
            --allow-overwrite=false \
            --file-allocation=none \
            --console-log-level=warn \
            --summary-interval=60 \
            --log="${LOG_DIR}/safetensors_aria2.log" \
            --log-level=notice \
            &
        pids+=($!)
        info "safetensors PID: ${pids[-1]}"
    fi

    if (( gguf_count > 0 )); then
        log "启动 GGUF 下载 (${gguf_count} 文件)..."
        aria2c \
            --input-file="${gguf_list}" \
            --max-concurrent-downloads=4 \
            --max-connection-per-server="${ARIA2_CONN}" \
            --split="${ARIA2_CONN}" \
            --min-split-size=10M \
            --max-tries="${ARIA2_RETRY}" \
            --retry-wait=5 \
            --timeout="${ARIA2_TIMEOUT}" \
            --connect-timeout=30 \
            --continue=true \
            --auto-file-renaming=false \
            --allow-overwrite=false \
            --file-allocation=none \
            --console-log-level=warn \
            --summary-interval=60 \
            --log="${LOG_DIR}/gguf_aria2.log" \
            --log-level=notice \
            &
        pids+=($!)
        info "GGUF PID: ${pids[-1]}"
    fi

    if (( ${#pids[@]} == 0 )); then
        log "所有模型文件已下载完成！"
        return 0
    fi

    log "下载进程已启动，后台运行中..."
    log "查看进度: bash $0 status"
    log "等待所有下载完成..."

    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            ((fail++))
        fi
    done

    echo ""
    if (( fail == 0 )); then
        log "========== 全部下载完成！ =========="
    else
        warn "有 ${fail} 个下载任务失败，请重新运行: bash $0 download"
    fi

    show_status
}

# ========================= 进度查看 =========================
show_status() {
    echo -e "\n${CYAN}==================== 下载状态 ====================${NC}"

    # safetensors
    local st_total=163
    local st_done=0
    local st_size="0"
    if [[ -d "${ST_DIR}" ]]; then
        st_done=$(find "${ST_DIR}" -name "*.safetensors" -size +1G 2>/dev/null | wc -l)
        st_size=$(du -sh "${ST_DIR}" 2>/dev/null | cut -f1)
    fi
    local st_pct=$((st_done * 100 / st_total))
    echo -e "${BLUE}[Safetensors]${NC} ${st_done}/${st_total} 文件 (${st_pct}%)  大小: ${st_size} / ~690G"

    # 进度条
    local bar_len=40
    local filled=$((st_pct * bar_len / 100))
    local empty=$((bar_len - filled))
    printf "  ["
    printf "%0.s█" $(seq 1 $filled 2>/dev/null) 2>/dev/null
    printf "%0.s░" $(seq 1 $empty 2>/dev/null) 2>/dev/null
    printf "]\n"

    # GGUF
    local gguf_total=9
    local gguf_done=0
    local gguf_size="0"
    if [[ -d "${GGUF_DIR}" ]]; then
        gguf_done=$(find "${GGUF_DIR}" -name "*.gguf" -size +1G 2>/dev/null | wc -l)
        gguf_size=$(du -sh "${GGUF_DIR}" 2>/dev/null | cut -f1)
    fi
    local gguf_pct=$((gguf_done * 100 / gguf_total))
    echo -e "${BLUE}[GGUF Q4_K_M]${NC}  ${gguf_done}/${gguf_total} 文件 (${gguf_pct}%)   大小: ${gguf_size} / ~378G"

    filled=$((gguf_pct * bar_len / 100))
    empty=$((bar_len - filled))
    printf "  ["
    printf "%0.s█" $(seq 1 $filled 2>/dev/null) 2>/dev/null
    printf "%0.s░" $(seq 1 $empty 2>/dev/null) 2>/dev/null
    printf "]\n"

    # 磁盘
    echo ""
    df -h /root/autodl-tmp 2>/dev/null | tail -1 | awk '{printf "  磁盘: 已用 %s / %s  可用 %s (%s)\n", $3, $2, $4, $5}'

    # aria2 进程
    local aria_procs
    aria_procs=$(pgrep -c aria2c 2>/dev/null || echo 0)
    if (( aria_procs > 0 )); then
        echo -e "\n  ${GREEN}▶ aria2c 正在运行 (${aria_procs} 个进程)${NC}"
    else
        if (( st_done == st_total && gguf_done == gguf_total )); then
            echo -e "\n  ${GREEN}✓ 全部下载完成！可以启动: bash $0 launch${NC}"
        else
            echo -e "\n  ${YELLOW}⏸ aria2c 未运行。如需继续: bash $0 download${NC}"
        fi
    fi
    echo -e "${CYAN}=================================================${NC}\n"
}

# ========================= 启动 KTransformers =========================
launch_kt() {
    log "========== 启动 KTransformers =========="

    # 检查文件完整性
    local st_done gguf_done
    st_done=$(find "${ST_DIR}" -name "*.safetensors" -size +1G 2>/dev/null | wc -l)
    gguf_done=$(find "${GGUF_DIR}" -name "*.gguf" -size +1G 2>/dev/null | wc -l)

    if (( st_done < 163 )); then
        err "safetensors 不完整: ${st_done}/163。请先完成下载: bash $0 download"
        exit 1
    fi
    if (( gguf_done < 9 )); then
        err "GGUF 不完整: ${gguf_done}/9。请先完成下载: bash $0 download"
        exit 1
    fi

    log "模型文件检查通过"
    log "  safetensors: ${ST_DIR} (${st_done} 文件)"
    log "  GGUF:        ${GGUF_DIR} (${gguf_done} 文件)"

    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"

    log "启动命令:"
    echo "  ktransformers \\"
    echo "    --model_path ${ST_DIR} \\"
    echo "    --gguf_path ${GGUF_DIR} \\"
    echo "    --host ${KT_HOST} \\"
    echo "    --port ${KT_PORT} \\"
    echo "    --cpu_infer ${KT_CPU_THREADS}"
    echo ""

    exec ktransformers \
        --model_path "${ST_DIR}" \
        --gguf_path "${GGUF_DIR}" \
        --host "${KT_HOST}" \
        --port "${KT_PORT}" \
        --cpu_infer "${KT_CPU_THREADS}"
}

# ========================= 主入口 =========================
main() {
    local cmd="${1:-all}"

    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════╗"
    echo "║  DeepSeek-V3.2 + KTransformers  AutoDL 一键部署  ║"
    echo "╚═══════════════════════════════════════════════════╝"
    echo -e "${NC}"

    case "${cmd}" in
        env)
            setup_env
            ;;
        download)
            download_all
            ;;
        status)
            show_status
            ;;
        launch)
            launch_kt
            ;;
        all)
            setup_env
            download_all
            ;;
        *)
            echo "用法: bash $0 {all|env|download|status|launch}"
            echo ""
            echo "  all      - 安装环境 + 下载模型（默认）"
            echo "  env      - 仅安装环境（conda + ktransformers）"
            echo "  download - 仅下载模型（safetensors + GGUF）"
            echo "  status   - 查看下载进度"
            echo "  launch   - 启动 KTransformers 服务"
            exit 1
            ;;
    esac
}

main "$@"
