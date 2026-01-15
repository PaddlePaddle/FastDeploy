package manager

import (
	"encoding/json"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v3"
)

func TestInstanceRole_String(t *testing.T) {
	assert.Equal(t, "mixed", MIXED.String())
	assert.Equal(t, "prefill", PREFILL.String())
	assert.Equal(t, "decode", DECODE.String())
}

func TestParseInstanceRole(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		expected  InstanceRole
		expectErr bool
	}{
		{"mixed lowercase", "mixed", MIXED, false},
		{"mixed uppercase", "MIXED", MIXED, false},
		{"prefill lowercase", "prefill", PREFILL, false},
		{"prefill uppercase", "PREFILL", PREFILL, false},
		{"decode lowercase", "decode", DECODE, false},
		{"decode uppercase", "DECODE", DECODE, false},
		{"invalid role", "invalid", -1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ParseInstanceRole(tt.input)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestRole_UnmarshalJSON(t *testing.T) {
	t.Run("valid role from integer", func(t *testing.T) {
		var role Role
		err := json.Unmarshal([]byte("0"), &role)
		assert.NoError(t, err)
		assert.Equal(t, MIXED, role.EnumValue)
		assert.False(t, role.IsCustom)
	})

	t.Run("valid role from string", func(t *testing.T) {
		var role Role
		err := json.Unmarshal([]byte(`"prefill"`), &role)
		assert.NoError(t, err)
		assert.Equal(t, PREFILL, role.EnumValue)
		assert.False(t, role.IsCustom)
	})

	t.Run("custom role", func(t *testing.T) {
		var role Role
		err := json.Unmarshal([]byte(`"custom-role"`), &role)
		assert.NoError(t, err)
		assert.Equal(t, "custom-role", role.CustomName)
		assert.True(t, role.IsCustom)
	})

	t.Run("invalid integer", func(t *testing.T) {
		var role Role
		err := json.Unmarshal([]byte("99"), &role)
		assert.Error(t, err)
	})
}

func TestRole_UnmarshalYAML(t *testing.T) {
	t.Run("valid role from integer", func(t *testing.T) {
		var role Role
		err := yaml.Unmarshal([]byte("1"), &role)
		assert.NoError(t, err)
		assert.Equal(t, PREFILL, role.EnumValue)
		assert.False(t, role.IsCustom)
	})

	t.Run("valid role from string", func(t *testing.T) {
		var role Role
		err := yaml.Unmarshal([]byte("decode"), &role)
		assert.NoError(t, err)
		assert.Equal(t, DECODE, role.EnumValue)
		assert.False(t, role.IsCustom)
	})
}

func TestRole_MarshalJSON(t *testing.T) {
	t.Run("standard role", func(t *testing.T) {
		role := Role{EnumValue: MIXED, IsCustom: false}
		data, err := json.Marshal(role)
		assert.NoError(t, err)
		assert.Equal(t, `"mixed"`, string(data))
	})

	t.Run("custom role", func(t *testing.T) {
		role := Role{CustomName: "custom", IsCustom: true}
		data, err := json.Marshal(role)
		assert.NoError(t, err)
		assert.Equal(t, `"custom"`, string(data))
	})
}

func TestPort_UnmarshalJSON(t *testing.T) {
	t.Run("port as integer", func(t *testing.T) {
		var port Port
		err := json.Unmarshal([]byte("8080"), &port)
		assert.NoError(t, err)
		assert.Equal(t, Port("8080"), port)
	})

	t.Run("port as string", func(t *testing.T) {
		var port Port
		err := json.Unmarshal([]byte(`"9090"`), &port)
		assert.NoError(t, err)
		assert.Equal(t, Port("9090"), port)
	})
}

func TestPort_UnmarshalYAML(t *testing.T) {
	t.Run("port as integer", func(t *testing.T) {
		var port Port
		err := yaml.Unmarshal([]byte("8080"), &port)
		assert.NoError(t, err)
		assert.Equal(t, Port("8080"), port)
	})
}

func TestIntToStringList_UnmarshalJSON(t *testing.T) {
	t.Run("mixed types", func(t *testing.T) {
		var list IntToStringList
		err := json.Unmarshal([]byte(`["1", 2, 3.0]`), &list)
		assert.NoError(t, err)
		assert.Equal(t, IntToStringList{"1", "2", "3"}, list)
	})

	t.Run("invalid float", func(t *testing.T) {
		var list IntToStringList
		err := json.Unmarshal([]byte(`[1.5]`), &list)
		assert.Error(t, err)
	})

	t.Run("invalid type", func(t *testing.T) {
		var list IntToStringList
		err := json.Unmarshal([]byte(`[true]`), &list)
		assert.Error(t, err)
	})
}

func TestInstanceInfo_URL(t *testing.T) {
	info := &InstanceInfo{
		HostIP: "127.0.0.1",
		Port:   Port("8080"),
	}

	url := info.URL()
	assert.Equal(t, "http://127.0.0.1:8080", url)
}

func TestNewInstanceInfo(t *testing.T) {
	// Setup DefaultManager
	Init(&config.Config{Server: config.ServerConfig{Splitwise: true}})

	t.Run("valid instance", func(t *testing.T) {
		info := &InstanceInfo{
			Role:             Role{EnumValue: PREFILL, IsSet: true},
			HostIP:           "127.0.0.1",
			Port:             Port("8080"),
			ConnectorPort:    Port("9000"),
			TransferProtocol: []string{"rdma"},
			RDMAPorts:        IntToStringList{"5000", "5001"},
			DeviceIDs:        IntToStringList{"0", "1"},
		}

		result, err := NewInstanceInfo(info)
		assert.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("missing role", func(t *testing.T) {
		info := &InstanceInfo{
			HostIP: "127.0.0.1",
			Port:   Port("8080"),
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "role is required")
	})

	t.Run("custom role", func(t *testing.T) {
		info := &InstanceInfo{
			Role:   Role{CustomName: "custom", IsCustom: true, IsSet: true},
			HostIP: "127.0.0.1",
			Port:   Port("8080"),
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid role")
	})

	t.Run("missing host_ip", func(t *testing.T) {
		info := &InstanceInfo{
			Role: Role{EnumValue: MIXED, IsSet: true},
			Port: Port("8080"),
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "host_ip is required")
	})

	t.Run("invalid host_ip", func(t *testing.T) {
		info := &InstanceInfo{
			Role:   Role{EnumValue: MIXED, IsSet: true},
			HostIP: "invalid-ip",
			Port:   Port("8080"),
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid host_ip")
	})

	t.Run("invalid port", func(t *testing.T) {
		info := &InstanceInfo{
			Role:   Role{EnumValue: MIXED, IsSet: true},
			HostIP: "127.0.0.1",
			Port:   Port("99999"),
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid port")
	})

	t.Run("invalid connector_port", func(t *testing.T) {
		info := &InstanceInfo{
			Role:          Role{EnumValue: PREFILL, IsSet: true},
			HostIP:        "127.0.0.1",
			Port:          Port("8080"),
			ConnectorPort: Port("99999"),
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid connector_port")
	})

	t.Run("invalid transfer protocol", func(t *testing.T) {
		info := &InstanceInfo{
			Role:             Role{EnumValue: PREFILL, IsSet: true},
			HostIP:           "127.0.0.1",
			Port:             Port("8080"),
			TransferProtocol: []string{"invalid"},
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid protocol")
	})

	t.Run("invalid rdma port", func(t *testing.T) {
		info := &InstanceInfo{
			Role:             Role{EnumValue: PREFILL, IsSet: true},
			HostIP:           "127.0.0.1",
			Port:             Port("8080"),
			TransferProtocol: []string{"rdma"},
			RDMAPorts:        IntToStringList{"99999"},
		}

		_, err := NewInstanceInfo(info)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "rdma_ports[0] invalid port")
	})
}

func TestIsValidPort(t *testing.T) {
	assert.True(t, isValidPort(Port("8080")))
	assert.True(t, isValidPort(Port("1")))
	assert.True(t, isValidPort(Port("65535")))
	assert.False(t, isValidPort(Port("0")))
	assert.False(t, isValidPort(Port("65536")))
	assert.False(t, isValidPort(Port("invalid")))
	assert.False(t, isValidPort(Port("")))
}

func TestIsValidIP(t *testing.T) {
	assert.True(t, isValidIP("127.0.0.1"))
	assert.True(t, isValidIP("192.168.1.1"))
	assert.True(t, isValidIP("::1"))
	assert.False(t, isValidIP("invalid"))
	assert.False(t, isValidIP(""))
}
