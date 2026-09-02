#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RENDERER="$SCRIPT_DIR/render_dataset.py"
PUPPET_EXPORTER="$SCRIPT_DIR/export_episode_json.py"
CAMERA_CAPTURE="$SCRIPT_DIR/capture_camera_info.py"
PYTHON="${PYTHON:-python}"
ROS2_PYTHON="${ROS2_PYTHON:-/usr/bin/python3}"

INPUT_DIR="/home/agilex/piper_dual_dataset/stack_cups_eef"
OUTPUT_DIR="/home/agilex/piper_dual_dataset/stack_cups_eef_rendered"
FPS=30
FORCE=false
MAKE_PLOTS=false

usage() {
    cat <<'EOF'
批量将 Piper dual HDF5 episode 渲染为视频，并导出 Puppet 状态与在线相机标定 JSON。

用法：
  render_hdf5_batch.sh [选项]

选项：
  --input-dir DIR   HDF5 输入目录
                    默认：/home/agilex/piper_dual_dataset/stack_cups_eef
  --output-dir DIR  视频和 JSON 输出根目录
                    默认：/home/agilex/piper_dual_dataset/stack_cups_eef_rendered
  --fps FPS         输出视频帧率，默认：30
  --plots           额外生成 state_action.png
  --force           强制重新渲染、导出 JSON 并重新读取 CameraInfo
  -h, --help        显示帮助

每个 episode 输出：
  cam_high.mp4、cam_left_wrist.mp4、cam_right_wrist.mp4、views_3x1.mp4
  quality.json、puppet_state.json、camera_parameters.json

camera_parameters.json 是批处理开始时从当前在线 ROS 2 CameraInfo 话题读取的快照，
不是历史 HDF5 内嵌标定证明。PYTHON 指定 HDF5/视频解释器，ROS2_PYTHON 指定 ROS 2 解释器（默认 /usr/bin/python3）。
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --input-dir)
            [[ $# -ge 2 ]] || { echo "错误：--input-dir 缺少参数" >&2; exit 2; }
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "错误：--output-dir 缺少参数" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fps)
            [[ $# -ge 2 ]] || { echo "错误：--fps 缺少参数" >&2; exit 2; }
            FPS="$2"
            shift 2
            ;;
        --plots)
            MAKE_PLOTS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误：未知参数：$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -d "$INPUT_DIR" ]] || { echo "错误：输入目录不存在：$INPUT_DIR" >&2; exit 1; }
[[ -f "$RENDERER" ]] || { echo "错误：找不到渲染器：$RENDERER" >&2; exit 1; }
[[ -f "$PUPPET_EXPORTER" ]] || { echo "错误：找不到 Puppet JSON 导出器：$PUPPET_EXPORTER" >&2; exit 1; }
[[ -f "$CAMERA_CAPTURE" ]] || { echo "错误：找不到 CameraInfo 采集器：$CAMERA_CAPTURE" >&2; exit 1; }
[[ "$FPS" =~ ^[1-9][0-9]*$ ]] || { echo "错误：--fps 必须是正整数：$FPS" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
hdf5_files=("$INPUT_DIR"/episode_*.hdf5)
if (( ${#hdf5_files[@]} == 0 )); then
    echo "错误：输入目录中没有 episode_*.hdf5 文件：$INPUT_DIR" >&2
    exit 1
fi

CAMERA_SNAPSHOT="$OUTPUT_DIR/.camera_parameters.json"
cleanup_camera_snapshot() {
    rm -f -- "$CAMERA_SNAPSHOT"
}
trap cleanup_camera_snapshot EXIT

install_camera_snapshot() {
    local episode_dir="$1"
    local destination="$episode_dir/camera_parameters.json"
    local temporary="$episode_dir/.camera_parameters.json.$$"

    mkdir -p -- "$episode_dir"
    rm -f -- "$temporary"
    if ! cp -- "$CAMERA_SNAPSHOT" "$temporary"; then
        rm -f -- "$temporary"
        return 1
    fi
    mv -f -- "$temporary" "$destination"
}


videos_complete() {
    local episode_dir="$1"
    local required=(
        cam_high.mp4
        cam_left_wrist.mp4
        cam_right_wrist.mp4
        views_3x1.mp4
        quality.json
    )
    local name

    if [[ "$MAKE_PLOTS" == true ]]; then
        required+=(state_action.png)
    fi
    for name in "${required[@]}"; do
        [[ -s "$episode_dir/$name" ]] || return 1
    done
    return 0
}

output_complete() {
    local episode_dir="$1"
    local required=(puppet_state.json camera_parameters.json)
    local name

    videos_complete "$episode_dir" || return 1
    for name in "${required[@]}"; do
        [[ -s "$episode_dir/$name" ]] || return 1
    done
    return 0
}

rendered=0
skipped=0
total=${#hdf5_files[@]}

echo "输入目录：$INPUT_DIR"
echo "输出目录：$OUTPUT_DIR"
echo "发现 $total 个 HDF5 文件，FPS=$FPS"
requires_work=false
for input_path in "${hdf5_files[@]}"; do
    episode="$(basename -- "$input_path" .hdf5)"
    if [[ "$FORCE" == true ]] || ! output_complete "$OUTPUT_DIR/$episode"; then
        requires_work=true
        break
    fi
done

if [[ "$requires_work" == true ]]; then
    echo "读取当前在线三路 CameraInfo..."
    "$ROS2_PYTHON" "$CAMERA_CAPTURE" --output "$CAMERA_SNAPSHOT"
    [[ -s "$CAMERA_SNAPSHOT" ]] || { echo "错误：CameraInfo 快照为空：$CAMERA_SNAPSHOT" >&2; exit 1; }
fi

for index in "${!hdf5_files[@]}"; do
    input_path="${hdf5_files[$index]}"
    filename="$(basename -- "$input_path")"
    episode="${filename%.hdf5}"
    episode_dir="$OUTPUT_DIR/$episode"

    if [[ "$FORCE" != true ]] && output_complete "$episode_dir"; then
        echo "[$((index + 1))/$total] 跳过完整结果：$episode"
        ((skipped += 1))
        continue
    fi

    echo "[$((index + 1))/$total] 正在处理：$episode"
    if [[ "$FORCE" == true ]] || ! videos_complete "$episode_dir"; then
        command=(
            "$PYTHON" "$RENDERER"
            --input "$input_path"
            --output-dir "$episode_dir"
            --fps "$FPS"
        )
        if [[ "$MAKE_PLOTS" != true ]]; then
            command+=(--no-plots)
        fi
        "${command[@]}"
    fi
    "$PYTHON" "$PUPPET_EXPORTER" \
        --input "$input_path" \
        --puppet-output "$episode_dir/puppet_state.json"
    install_camera_snapshot "$episode_dir"
    output_complete "$episode_dir" || {
        echo "错误：episode 输出不完整：$episode_dir" >&2
        exit 1
    }
    ((rendered += 1))
done

echo "批量渲染完成：共 $total，生成 $rendered，跳过 $skipped。"
