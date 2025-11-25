# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

FD_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/FastDeploy/pulls/${PR_ID}/reviews?per_page=10000`
failed_num=0
echo_list=()


function check_approval(){
    local echo_line="$1"
    shift
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${FD_ROOT}/scripts/check_pr_approval.py $1 $person_num`
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}


function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}


HAS_CUSTOM_REGISTRER=`git diff -U0 upstream/$BRANCH | grep '^\+' | grep -zoE "PD_BUILD_(STATIC_)?OP" || true`
if [ ${HAS_CUSTOM_REGISTRER} ] && [ "${PR_ID}" != "" ]; then
    echo_line1="You must have one FastDeploy RD (qingqing01(dangqingqing), Jiang-Jia-Jun(jiangjiajun), heavengate(dengkaipeng)) approval for adding custom op.\n"
    echo_line2="You must have one QA(DDDivano(zhengtianyu)) approval for adding custom op.\n"
    echo_line3="You must have one PaddlePaddle RD (XiaoguangHu01(huxiaoguang), jeff41404(gaoxiang), phlrain(liuhongyu)) approval for adding custom op.\n"
    check_approval "$echo_line1" 1 qingqing01 Jiang-Jia-Jun heavengate
    check_approval "$echo_line2" 1 DDDivano
    check_approval "$echo_line3" 1 XiaoguangHu01 jeff41404 phlrain
fi

WORKER_OR_CONFIG_LIST=(
    "fastdeploy/config.py"
    "fastdeploy/worker"
    "fastdeploy/model_executor/graph_optimization"
    "fastdeploy/model_executor/model_loader"
    "fastdeploy/model_executor/models"
)

HAS_WORKER_OR_CONFIG_MODIFY=`git diff upstream/$BRANCH  --name-only | grep -E $(printf -- "-e %s " "${WORKER_OR_CONFIG_LIST[@]}") || true`
if [ "${HAS_WORKER_OR_CONFIG_MODIFY}" != "" ] && [ "${PR_ID}" != "" ]; then
    echo_line1="You must have one FastDeploy RD (gongshaotian(gongshaotian), yuanlehome(liuyuanle)) approval for modifing [$(IFS=', '; echo "${WORKER_OR_CONFIG_LIST[*]}")]."
    check_approval "$echo_line1" 1 gongshaotian yuanlehome
fi

SPECULATIVE_DECODING_LIST=(
    "fastdeploy/spec_decode"
    "custom_ops/gpu_ops/speculate_decoding"
)

HAS_SPECULATIVE_DECODING_MODIFY=`git diff upstream/$BRANCH  --name-only | grep -E $(printf -- "-e %s " "${SPECULATIVE_DECODING_LIST[@]}") || true`
if [ "${HAS_SPECULATIVE_DECODING_MODIFY}" != "" ] && [ "${PR_ID}" != "" ]; then
    echo_line1="You must have one FastDeploy RD (freeliuzc(liuzichang01), Deleter-D(wangyanpeng04)) approval for modifing [$(IFS=', '; echo "${SPECULATIVE_DECODING_LIST[*]}")]."
    check_approval "$echo_line1" 1 freeliuzc Deleter-D
fi

if [[ "${BRANCH}" != "develop" ]] && [[ -n "${PR_ID}" ]]; then
    pr_info=$(curl -H "Authorization: token ${GITHUB_TOKEN}" \
        https://api.github.com/repos/PaddlePaddle/FastDeploy/pulls/${PR_ID})

    pr_title=$(echo "$pr_info" | jq -r '.title')
    echo "==> PR title: ${pr_title}"

    has_cp_tag=$(echo "$pr_title" | grep -o "\[Cherry-Pick\]" || true)
    has_pr_number=$(echo "$pr_title" | grep -oE "#[0-9]{2,6}" || true)

    if [[ -z "$has_cp_tag" ]] || [[ -z "$has_pr_number" ]]; then
        echo_line="Cherry-Pick PR must come from develop and the title must contain [Cherry-Pick] and the original develop PR number (e.g., #5010). Approval required from FastDeploy RD: qingqing01(dangqingqing), Jiang-Jia-Jun(jiangjiajun), heavengate(dengkaipeng)."
        check_approval "$echo_line" 1 qingqing01 Jiang-Jia-Jun heavengate
    fi
fi

if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are ${failed_num} approved errors."
  echo "****************"
fi

if [ -n "${echo_list}" ]; then
  exit 6
fi
