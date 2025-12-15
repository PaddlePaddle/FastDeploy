#!/bin/bash

# It is used to get the RDMA NICs
# This file is same as the one in fastdeploy/scripts/get_rdma_nics.sh

Cur_Dir=$(cd `dirname $0`; pwd)
NICNAME_TYPE=xgbe  # 默认检测类型
type=$1

if [ "$ENABLE_EP_DP" == "1" ]; then
    gpu_root_port_filename="${Cur_Dir}/gpu_rootport_${DP_RANK}.txt"
else
    gpu_root_port_filename="${Cur_Dir}/gpu_rootport.txt"
fi

function __NEW_GPU_ROOTPORT_FILE__() {
    touch ${gpu_root_port_filename} 2>/dev/null
    echo "" > ${gpu_root_port_filename} 2>/dev/null
    for gpu_bus in $(lspci 2>/dev/null | grep -iE "Communication controller: | controller: NVIDIA" | awk '{print $1}')
    do
        readlink "/sys/bus/pci/devices/0000:${gpu_bus}" 2>/dev/null | awk -F [/] '{print $6}' >> ${gpu_root_port_filename}
    done
}

function  __RM_GPU_ROOTPORT_FILE__() {
    rm -rf ${gpu_root_port_filename} 2>/dev/null
}

function __JUDGE_NIC_TYPE__() {
    XGBE_NUM=$(ip a 2>/dev/null | grep -c ": ${NICNAME_TYPE}")
    gpu_first=true
    xpu_first=true
    cpu_first=true

    for (( xgbe_no=0; xgbe_no < XGBE_NUM; xgbe_no++ ))
    do
        [ ! -d "/sys/class/net/${NICNAME_TYPE}${xgbe_no}" ] && continue

        PCI_ADDRESS=$(ethtool -i "${NICNAME_TYPE}${xgbe_no}" 2>/dev/null | awk -F '0000:' '/bus-info/{print $2}')
        [ -z "$PCI_ADDRESS" ] && continue
        NIC_ROOTPORT=$(readlink "/sys/bus/pci/devices/0000:${PCI_ADDRESS}" 2>/dev/null | awk -F '/' '{print $6}')

        NIC_TYPE="CPU_NIC"
        grep -qxF "$NIC_ROOTPORT" ${gpu_root_port_filename} 2>/dev/null && NIC_TYPE="GPU_NIC"

        if [[ "$type" == "gpu" && "$NIC_TYPE" == "GPU_NIC" ]]; then
            ibdev=$(ibdev2netdev 2>/dev/null | awk -v nic="${NICNAME_TYPE}${xgbe_no}" '$5 == nic {print $1}')
            if [ -n "$ibdev" ] && ip link show "${NICNAME_TYPE}${xgbe_no}" | grep -q "state UP"; then
                if $gpu_first; then
                    printf "KVCACHE_RDMA_NICS=%s" "$ibdev"
                    gpu_first=false
                else
                    printf ",%s" "$ibdev"
                fi
            fi
        fi

        if [[ "$type" == "xpu" && "$NIC_TYPE" == "GPU_NIC" ]]; then
            ibdev=$(ibdev2netdev 2>/dev/null | awk -v nic="${NICNAME_TYPE}${xgbe_no}" '$5 == nic {print $1}')
            if [ -n "$ibdev" ] && ip link show "${NICNAME_TYPE}${xgbe_no}" | grep -q "state UP"; then
                if $xpu_first; then
                    printf "KVCACHE_RDMA_NICS=%s,%s" "$ibdev" "$ibdev"
                    xpu_first=false
                else
                    printf ",%s,%s" "$ibdev" "$ibdev"
                fi
            fi
        fi

        if [[ "$type" == "cpu" ]]; then
            for (( xgbe_no=0; xgbe_no < XGBE_NUM; xgbe_no++ ))
            do
                [ ! -d "/sys/class/net/${NICNAME_TYPE}${xgbe_no}" ] && continue

                PCI_ADDRESS=$(ethtool -i "${NICNAME_TYPE}${xgbe_no}" 2>/dev/null | awk -F '0000:' '/bus-info/{print $2}')
                [ -z "$PCI_ADDRESS" ] && continue

                NIC_ROOTPORT=$(readlink "/sys/bus/pci/devices/0000:${PCI_ADDRESS}" 2>/dev/null | awk -F '/' '{print $6}')
                grep -qxF "$NIC_ROOTPORT" ${gpu_root_port_filename} 2>/dev/null && continue

                if ip link show "${NICNAME_TYPE}${xgbe_no}" | grep -q "state UP" && \
                ip a show "${NICNAME_TYPE}${xgbe_no}" | grep -q "inet"; then
                    printf "KV_CACHE_SOCKET_IFNAME=%s\n" "${NICNAME_TYPE}${xgbe_no}"
                    return 0
                fi
            done
                echo "ERROR: No active CPU NIC with IP found!" >&2
                return 1
        fi

        if [[ "$type" == "cpu_ib" && "$NIC_TYPE" == "CPU_NIC" ]]; then
            ibdev=$(ibdev2netdev 2>/dev/null | awk -v nic="${NICNAME_TYPE}${xgbe_no}" '$5 == nic {print $1}')
            if [ -n "$ibdev" ] && ip link show "${NICNAME_TYPE}${xgbe_no}" | grep -q "state UP" && \
               ip a show "${NICNAME_TYPE}${xgbe_no}" | grep -q "inet "; then
                if $cpu_ib_first; then
                    printf "KVCACHE_RDMA_NICS=%s" "$ibdev"
                    cpu_ib_first=false
                else
                    printf ",%s" "$ibdev"
                fi
            fi
        fi

    done

    case "$type" in
        gpu) ! $gpu_first && printf "\n" ;;
        xpu) ! $xpu_first && printf "\n" ;;
        cpu) ! $cpu_first && printf "\n" ;;
        cpu_ib) ! $cpu_ib_first && printf "\n" ;;
    esac
}

function get_vxpu_nics() {
    local topo_output=$(xpu-smi topo -m)
    local xpu_info=$(echo "$topo_output" | grep -E '^XPU[0-9]+')

    local nic_mapping=()
    while IFS= read -r line; do
        if [[ $line =~ NIC([0-9]+):\ +(mlx[0-9_]+) ]]; then
            local nic_idx=${BASH_REMATCH[1]}
            local nic_name=${BASH_REMATCH[2]}
            nic_mapping[$nic_idx]=$nic_name
        fi
    done < <(echo "$topo_output" | grep -E '^\s*NIC[0-9]+:')

    local nic_count=${#nic_mapping[@]}

    declare -A priority_map=([PIX]=2 [NODE]=1 [SYS]=0)
    local optimal_nics=()

    while IFS= read -r line; do
        local fields=($line)
        local nic_start_index=5
        local max_nics=$(( ${#fields[@]} - nic_start_index ))
        local actual_nic_count=$(( max_nics < nic_count ? max_nics : nic_count ))

        local best_priority=-1
        local best_nic=""

        for ((nic_idx=0; nic_idx<actual_nic_count; nic_idx++)); do
            local conn_type=${fields[nic_idx+nic_start_index]}
            local current_priority=${priority_map[$conn_type]:--1}

            if (( current_priority > best_priority )); then
                best_priority=$current_priority
                best_nic="${nic_mapping[$nic_idx]}"
            fi
        done

        if [[ -n "$best_nic" ]]; then
            optimal_nics+=("$best_nic")
        fi
    done <<< "$xpu_info"

    local IFS=,
    export KVCACHE_RDMA_NICS="${optimal_nics[*]}"
    echo "KVCACHE_RDMA_NICS=${optimal_nics[*]}"
}

function get_vcpu_nics() {
    ip -o addr show | awk '$3 == "inet" && $4 ~ /^10\./ {print "KV_CACHE_SOCKET_IFNAME="$2; exit}'
}

function __main__() {
    if [[ "$type" == "vxpu" ]]; then
        get_vxpu_nics
        return 0
    fi
    if [[ "$type" == "vcpu" ]]; then
        get_vcpu_nics
        return 0
    fi

    # 处理 bond 情况
    if [[ "$type" == "cpu" ]]; then
        for bond in $(ls -d /sys/class/net/bond* 2>/dev/null); do
            bond_if=$(basename "$bond")
            if ip link show "$bond_if" | grep -q "state UP" && \
               ip a show "$bond_if" | grep -q "inet "; then
                printf "KV_CACHE_SOCKET_IFNAME=%s\n" "$bond_if"
                return 0
            fi
        done
    fi

    if [[ "$type" == "cpu_ib" ]]; then
        first=true
        for bond in $(ls -d /sys/class/net/bond* 2>/dev/null); do
            bond_if=$(basename "$bond")
            __NEW_GPU_ROOTPORT_FILE__

            ibdev=$(ibdev2netdev 2>/dev/null | grep -w "$bond_if" | awk '{print $1}')
            if [ -n "$ibdev" ] && ip link show "$bond_if" | grep -q "state UP" && \
               ip a show "$bond_if" | grep -q "inet "; then
                if $first; then
                    printf "KVCACHE_RDMA_NICS=%s" "$ibdev"
                    first=false
                else
                    printf ",%s" "$ibdev"
                fi
            fi

            bondib=$(show_gids 2>/dev/null | grep -w "$bond_if" | awk '{print $1}' | grep "mlx.*bond" | head -1)
            if [ -n "$bondib" ] && ip link show "$bond_if" | grep -q "state UP" && \
               ip a show "$bond_if" | grep -q "inet " && $first; then
                printf "KVCACHE_RDMA_NICS=%s" "$bondib"
                first=false
            fi

            __RM_GPU_ROOTPORT_FILE__
        done

        ! $first && printf "\n"
        [ ! $first ] && return 0
    fi

    local nic_types=("eth" "ib" "xgbe")
    for nt in "${nic_types[@]}"; do
        if ip a | grep -iq "$nt"; then
            __NEW_GPU_ROOTPORT_FILE__
            NICNAME_TYPE=$nt
            __JUDGE_NIC_TYPE__
            __RM_GPU_ROOTPORT_FILE__
        fi
    done
}

__main__
