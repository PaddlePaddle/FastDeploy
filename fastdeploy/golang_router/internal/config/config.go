package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server    ServerConfig    `yaml:"server"`
	Log       LogConfig       `yaml:"log"`
	Manager   ManagerConfig   `yaml:"manager"`
	Scheduler SchedulerConfig `yaml:"scheduler"`
}

type ServerConfig struct {
	Name      string `yaml:"name"`
	Port      string `yaml:"port"`
	Host      string `yaml:"host"`
	Mode      string `yaml:"mode"` // debug, release, test
	Splitwise bool   `yaml:"splitwise"`
}

type ManagerConfig struct {
	RegisterPath            string  `yaml:"register-path"`
	HealthFailureThreshold  int     `yaml:"health-failure-threshold"`
	HealthSuccessThreshold  int     `yaml:"health-success-threshold"`
	HealthCheckTimeoutSecs  float64 `yaml:"health-check-timeout-secs"`
	HealthCheckIntervalSecs float64 `yaml:"health-check-interval-secs"`
	HealthCheckEndpoint     string  `yaml:"health-check-endpoint"`
}

type SchedulerConfig struct {
	Policy               string  `yaml:"policy"`
	PrefillPolicy        string  `yaml:"prefill-policy"`
	DecodePolicy         string  `yaml:"decode-policy"`
	EvictionIntervalSecs float64 `yaml:"eviction-interval-secs"`
	CacheBlockSize       int     `yaml:"cache-block-size"`
	TokenizerURL         string  `yaml:"tokenizer-url"`
	TokenizerTimeoutSecs float64 `yaml:"tokenizer-timeout-secs"`
	BalanceAbsThreshold  float64 `yaml:"balance-abs-threshold"`
	BalanceRelThreshold  float64 `yaml:"balance-rel-threshold"`
	HitRatioWeight       float64 `yaml:"hit-ratio-weight"`
	LoadBalanceWeight    float64 `yaml:"load-balance-weight"`
	WaitingWeight        float64 `yaml:"waiting-weight"`
}

type LogConfig struct {
	Level  string `yaml:"level"`  // debug, info, warn, error
	Output string `yaml:"output"` // stdout, file
}

func Load(configPath, listenPort string, isSplitwise bool) (*Config, error) {
	var cfg Config
	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}

		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse config: %w", err)
		}
	}

	// Set default values
	if listenPort != "" {
		cfg.Server.Port = listenPort
	} else if cfg.Server.Port == "" {
		return nil, fmt.Errorf("failed to set router listen port")
	}
	if isSplitwise {
		cfg.Server.Splitwise = true
	}
	if cfg.Server.Mode == "" {
		cfg.Server.Mode = "release"
	}
	if cfg.Log.Level == "" {
		cfg.Log.Level = "info"
	}
	if cfg.Manager.HealthCheckEndpoint == "" {
		cfg.Manager.HealthCheckEndpoint = "/health"
	}
	if cfg.Manager.HealthCheckTimeoutSecs == 0 {
		cfg.Manager.HealthCheckTimeoutSecs = 5
	}
	if cfg.Manager.HealthCheckIntervalSecs == 0 {
		cfg.Manager.HealthCheckIntervalSecs = 5
	}
	if cfg.Manager.HealthFailureThreshold == 0 {
		cfg.Manager.HealthFailureThreshold = 1
	}
	if cfg.Manager.HealthSuccessThreshold == 0 {
		cfg.Manager.HealthSuccessThreshold = 1
	}
	if cfg.Scheduler.EvictionIntervalSecs == 0 {
		cfg.Scheduler.EvictionIntervalSecs = 60
	}
	if cfg.Scheduler.CacheBlockSize == 0 {
		cfg.Scheduler.CacheBlockSize = 64
	}
	if cfg.Scheduler.TokenizerTimeoutSecs == 0 {
		cfg.Scheduler.TokenizerTimeoutSecs = 2
	}
	if cfg.Scheduler.HitRatioWeight == 0 {
		cfg.Scheduler.HitRatioWeight = 1
	}
	if cfg.Scheduler.LoadBalanceWeight == 0 {
		cfg.Scheduler.LoadBalanceWeight = 1
	}
	if cfg.Scheduler.BalanceAbsThreshold == 0 {
		cfg.Scheduler.BalanceAbsThreshold = 1
	}
	if cfg.Scheduler.BalanceRelThreshold == 0 {
		cfg.Scheduler.BalanceRelThreshold = 0.2
	}
	if cfg.Scheduler.WaitingWeight == 0 {
		cfg.Scheduler.WaitingWeight = 1
	}
	return &cfg, nil
}
