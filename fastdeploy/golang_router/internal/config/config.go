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
	Policy              string  `yaml:"policy"`
	PrefillPolicy       string  `yaml:"prefill-policy"`
	DecodePolicy        string  `yaml:"decode-policy"`
	IntervalCleanupSecs float64 `yaml:"interval-cleanup-secs"`
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
	if cfg.Scheduler.IntervalCleanupSecs == 0 {
		cfg.Scheduler.IntervalCleanupSecs = 60
	}
	return &cfg, nil
}
