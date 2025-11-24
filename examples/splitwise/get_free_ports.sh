function is_port_free() {
  local port=$1
  if ss -ltn | awk '{print $4}' | grep -q ":${port}$"; then
    return 1  # 占用
  fi
  return 0  # 空闲
}


function get_free_port() {
  free_ports_num=$1
  start_port=$2
  end_port=$3

  free_ports=()
  if [[ ! -n ${free_ports_num} || "${free_ports_num}" -le 0 ]]; then
    log_warn "param can't be empty, and should > 0"
    echo ${free_ports[@]}
    return 1
  fi

  used_ports1=$(netstat -an | grep -E "(0.0.0.0|127.0.0.1|${POD_IP}|tcp6)" | awk '{n=split($4,a,":"); if(a[n]~/^[0-9]+$/) print a[n];}' | sort -u)
  used_ports2=$(netstat -an | grep -E "(0.0.0.0|127.0.0.1|${POD_IP}|tcp6)" | awk '{n=split($5,a,":"); if(a[n]~/^[0-9]+$/) print a[n];}' | sort -u)
  all_used_ports=$(printf "%s\n" "${used_ports1}" "${used_ports2}" | sort -u)

  # 生成0到32767之间的随机数
  random_num=$(( RANDOM ))
  port=$(( random_num % (end_port - start_port + 1) + start_port ))

  while true; do
    (( port++ ))
    if [[ ${port} -ge ${end_port} ]]; then
      break
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

  echo ${free_ports[@]}
  return 0
}

free_ports=($(get_free_port ${1:-1} 8000 9000))

IFS=',' && echo "${free_ports[*]}"
# for ((i=0; i< $1; i++))
# do
#     echo ${free_ports[(($i))]}
# done
