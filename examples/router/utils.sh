#!/bin/bash

is_port_free() {
  local port=$1
  if ss -ltun | awk '{print $4}' | grep -q ":${port}$"; then
    return 1  # Port is occupied
  fi
  return 0  # Port is free
}

check_ports() {
    for port in "$@"; do
        if ! is_port_free $port; then
            echo "❌ Port $port is already in use"
            return 1
        fi
    done
    return 0
}

wait_for_health() {
    IFS=',' read -r -a server_ports <<< "$1"
    local num_ports=${#server_ports[@]}
    local total_lines=$((num_ports + 1))
    local first_run=true
    local GREEN='\033[0;32m'
    local RED='\033[0;31m'
    local NC='\033[0m' # No Color
    local start_time=$(date +%s)

    echo "-------- WAIT FOR HEALTH --------"
    while true; do
        local all_ready=true
        for port in "${server_ports[@]}"; do
            status_code=$(curl -s --max-time 1 -o /dev/null -w "%{http_code}" "http://0.0.0.0:${port}/health" || echo "000")
            if [ "$status_code" -eq 200 ]; then
                printf "Port %s: ${GREEN}[OK]   200${NC}\033[K\n" "$port"
            else
                all_ready=false
                printf "Port %s: ${RED}[WAIT] %s${NC}\033[K\n" "$port" "$status_code"
            fi
        done
        cur_time=$(date +%s)
        if [ "$all_ready" = "true" ]; then
            echo "All services are ready!    [$((cur_time-start_time))s]"
            break
        else
            echo "Services not ready..       [$((cur_time-start_time))s]"
            printf "\033[%dA" "$total_lines"  # roll back cursor
            sleep 1
        fi
    done
    echo "---------------------------------"
}

get_free_ports() {
  free_ports_num=${1:-1}
  start_port=${2:-8000}
  end_port=${3:-9000}

  free_ports=()
  if [[ ! -n ${free_ports_num} || "${free_ports_num}" -le 0 ]]; then
    log_warn "param can't be empty, and should > 0"
    echo ${free_ports[@]}
    return 1
  fi

  used_ports1=$(netstat -an | grep -E "(0.0.0.0|127.0.0.1|${POD_IP}|tcp6)" | awk '{n=split($4,a,":"); if(a[n]~/^[0-9]+$/) print a[n];}' | sort -u)
  used_ports2=$(netstat -an | grep -E "(0.0.0.0|127.0.0.1|${POD_IP}|tcp6)" | awk '{n=split($5,a,":"); if(a[n]~/^[0-9]+$/) print a[n];}' | sort -u)
  all_used_ports=$(printf "%s\n" "${used_ports1}" "${used_ports2}" | sort -u)

  # Generate random number between 0 and 32767
  random_num=$(( RANDOM ))
  port=$(( random_num % (end_port - start_port + 1) + start_port ))

  while true; do
    (( port++ ))
    if [[ ${port} -ge ${end_port} ]]; then
      port=${start_port}
    fi

    if [[ "${all_used_ports[@]}" =~ "${port}" ]]; then
      continue
    fi

    if is_port_free ${port}; then
      free_ports+=("${port}")
      (( free_ports_num-- ))
      if [[ ${free_ports_num} = 0 ]]; then
        break
      fi
    fi

  done

  # echo ${free_ports[@]}
  IFS=',' && echo "${free_ports[*]}"
  return 0
}
