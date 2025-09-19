#!/bin/bash
# 运行位置：~/Desktop/ML_PDE/RFM/pyRFM/test

set -euo pipefail

PROJECT_ID=945674
IMAGE="registry.dp.tech/dptech/dp/native/prod-17893/pyrfm:surface"
MACHINE="c6_m60_1 * NVIDIA 4090"   # 完整机型名要加引号
INPUT_DIR="surface"
WORKDIR="surface/sec3/sec3_2"
RESULT_BASE="/personal/results/pyRFM/sec3_2"

# 1) 创建任务组
GROUP_NAME="rfm-sec3_2-$(date +%Y%m%d-%H%M%S)"
echo "Creating job group: ${GROUP_NAME}"
JG_OUT=$(bohr job_group create -n "${GROUP_NAME}" -p ${PROJECT_ID} 2>&1)

echo "Raw output:"
echo "${JG_OUT}"

# 解析 job_group_id（兼容小写/大小写、信息行前后有文字等情况）
JOB_GROUP_ID=$(echo "${JG_OUT}" | sed -n 's/.*job[_ ]*group[_ ]*id:[[:space:]]*\([0-9]\+\).*/\1/p')
if [ -z "${JOB_GROUP_ID}" ]; then
  # 兜底：抓第一串纯数字（若整行只包含一个 ID）
  JOB_GROUP_ID=$(echo "${JG_OUT}" | grep -Eo '[0-9]+' | head -n1 || true)
fi

if [ -z "${JOB_GROUP_ID}" ]; then
  echo "❌ Failed to parse job_group_id. Please check the above output."
  exit 1
fi
echo "✅ JobGroupId = ${JOB_GROUP_ID}"

# 2) 批量提交 *_in.pth
shopt -s nullglob
PTHS=(${INPUT_DIR}/data/*_in.pth)
if [ ${#PTHS[@]} -eq 0 ]; then
  echo "No *_in.pth found under ${INPUT_DIR}/data/"
  exit 1
fi

for pth in "${PTHS[@]}"; do
  name=$(basename "$pth" .pth)    # e.g. bunny_in
  short=${name%_in}               # e.g. bunny
  echo "Submitting job for ${name} (-> ${short}) ..."

  bohr job submit \
    -m "${IMAGE}" \
    -t "${MACHINE}" \
    -p "${INPUT_DIR}" \
    -n "rfm-${short}" \
    -c "pip install pyrfm==0.2.5 torch numpy scipy matplotlib typing_extensions pyyaml scikit-image && cd ${WORKDIR} && python train_sdf.py --pth_path ../../data/${name}.pth" \
    --project_id ${PROJECT_ID} \
    -g ${JOB_GROUP_ID} \
    -r "${RESULT_BASE}/${short}"
done

echo "🎉 All jobs submitted into group: ${GROUP_NAME} (ID: ${JOB_GROUP_ID})"