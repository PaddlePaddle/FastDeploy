package manager

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"slices"
	"strconv"
	"strings"
)

type InstanceRole int

const (
	MIXED InstanceRole = iota
	PREFILL
	DECODE
)

var roleNames = [...]string{"mixed", "prefill", "decode"}

func (r InstanceRole) String() string { return roleNames[r] }

func ParseInstanceRole(s string) (InstanceRole, error) {
	for i, name := range roleNames {
		if strings.EqualFold(strings.ToLower(s), name) {
			return InstanceRole(i), nil
		}
	}
	return -1, fmt.Errorf("invalid role: %s", s)
}

type Role struct {
	EnumValue  InstanceRole
	CustomName string
	IsCustom   bool
	IsSet      bool
}

func (r *Role) parse(getInt func() (int, error), getStr func() (string, error)) error {
	r.IsSet = true
	if i, err := getInt(); err == nil {
		if i >= 0 && i <= int(DECODE) {
			r.EnumValue, r.IsCustom = InstanceRole(i), false
			return nil
		}
		return fmt.Errorf("invalid role integer: %d", i)
	}
	s, err := getStr()
	if err != nil {
		return err
	}
	if e, err := ParseInstanceRole(s); err == nil {
		r.EnumValue, r.IsCustom = e, false
	} else {
		r.CustomName, r.IsCustom = s, true
	}
	return nil
}

func (r *Role) UnmarshalJSON(data []byte) error {
	return r.parse(
		func() (int, error) { var i int; return i, json.Unmarshal(data, &i) },
		func() (string, error) { var s string; return s, json.Unmarshal(data, &s) },
	)
}

func (r *Role) UnmarshalYAML(u func(interface{}) error) error {
	return r.parse(
		func() (int, error) { var i int; return i, u(&i) },
		func() (string, error) { var s string; return s, u(&s) },
	)
}

func (r Role) MarshalJSON() ([]byte, error) {
	if r.IsCustom {
		return json.Marshal(r.CustomName)
	}
	return json.Marshal(r.EnumValue.String())
}

type Port string

func (p *Port) UnmarshalJSON(data []byte) error {
	var i int
	if json.Unmarshal(data, &i) == nil {
		*p = Port(strconv.Itoa(i))
		return nil
	}
	return json.Unmarshal(data, (*string)(p))
}

func (p *Port) UnmarshalYAML(u func(interface{}) error) error {
	var i int
	if u(&i) == nil {
		*p = Port(strconv.Itoa(i))
		return nil
	}
	return u((*string)(p))
}

type IntToStringList []string

func (sl *IntToStringList) UnmarshalJSON(data []byte) error {
	return sl.unmarshal(data, json.Unmarshal)
}
func (sl *IntToStringList) UnmarshalYAML(u func(interface{}) error) error {
	return sl.unmarshal(nil, func(_ []byte, v interface{}) error { return u(v) })
}

func (sl *IntToStringList) unmarshal(data []byte, u func([]byte, interface{}) error) error {
	var raw []interface{}
	if err := u(data, &raw); err != nil {
		return err
	}
	res := make([]string, len(raw))
	for i, v := range raw {
		switch val := v.(type) {
		case string:
			res[i] = val
		case int:
			res[i] = strconv.Itoa(val)
		case float64:
			if val == float64(int(val)) {
				res[i] = strconv.Itoa(int(val))
			} else {
				return fmt.Errorf("element %d: %v not integer", i, val)
			}
		default:
			return fmt.Errorf("element %d: type %T unsupported", i, v)
		}
	}
	*sl = res
	return nil
}

type InstanceInfo struct {
	Role                  Role            `json:"role" yaml:"role"`
	HostIP                string          `json:"host_ip" yaml:"host_ip"`
	Port                  Port            `json:"port" yaml:"port"`
	ConnectorPort         Port            `json:"connector_port,omitempty" yaml:"connector_port,omitempty"`
	EngineWorkerQueuePort Port            `json:"engine_worker_queue_port,omitempty" yaml:"engine_worker_queue_port,omitempty"`
	TransferProtocol      []string        `json:"transfer_protocol,omitempty" yaml:"transfer_protocol,omitempty"`
	RDMAPorts             IntToStringList `json:"rdma_ports,omitempty" yaml:"rdma_ports,omitempty"`
	DeviceIDs             IntToStringList `json:"device_ids,omitempty" yaml:"device_ids,omitempty"`
	MetricsPort           Port            `json:"metrics_port,omitempty" yaml:"metrics_port,omitempty"`
}

func isValidPort(p Port) bool {
	i, err := strconv.Atoi(string(p))
	if err != nil {
		return false
	}
	return i > 0 && i <= 65535
}

func isValidIP(ip string) bool {
	return net.ParseIP(ip) != nil
}

func validatePortList(name string, list []string) error {
	for i, p := range list {
		portInt, err := strconv.Atoi(p)
		if err != nil || portInt <= 0 || portInt > 65535 {
			return fmt.Errorf("%s[%d] invalid port: %s", name, i, p)
		}
	}
	return nil
}

func (info *InstanceInfo) URL() string {
	url := fmt.Sprintf("%s:%s", info.HostIP, info.Port)
	if !strings.HasPrefix(url, "http") {
		url = "http://" + url
	}
	return url
}

func NewInstanceInfo(info *InstanceInfo) (*InstanceInfo, error) {
	if !info.Role.IsSet {
		return nil, errors.New("role is required")
	}
	if info.Role.IsCustom {
		return nil, fmt.Errorf("invalid role: %s", info.Role.CustomName)
	}
	if info.HostIP == "" {
		return nil, errors.New("host_ip is required")
	}
	if !isValidIP(info.HostIP) {
		return nil, fmt.Errorf("invalid host_ip: %s", info.HostIP)
	}
	if info.Port == "" {
		return nil, errors.New("port is required")
	}
	if !isValidPort(info.Port) {
		return nil, fmt.Errorf("invalid port: %s", info.Port)
	}
	if DefaultManager.splitwise && info.ConnectorPort != "" && !isValidPort(info.ConnectorPort) {
		return nil, fmt.Errorf("invalid connector_port: %s", info.ConnectorPort)
	}
	if DefaultManager.splitwise && info.EngineWorkerQueuePort != "" && !isValidPort(info.EngineWorkerQueuePort) {
		return nil, fmt.Errorf("invalid engine_worker_queue_port: %s", info.EngineWorkerQueuePort)
	}
	for _, proto := range info.TransferProtocol {
		if !slices.Contains([]string{"ipc", "rdma"}, proto) {
			return nil, fmt.Errorf("invalid protocol: %s", proto)
		}
	}
	if err := validatePortList("rdma_ports", info.RDMAPorts); DefaultManager.splitwise && err != nil {
		return nil, err
	}
	if info.MetricsPort == "" {
		info.MetricsPort = info.Port
	} else {
		if !isValidPort(info.MetricsPort) {
			return nil, fmt.Errorf("invalid metrics_port: %s", info.MetricsPort)
		}
	}
	return info, nil
}
