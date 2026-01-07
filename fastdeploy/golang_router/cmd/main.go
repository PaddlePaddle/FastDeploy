package main

import (
	"context"
	"flag"
	"log"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/PaddlePaddle/FastDeploy/router/internal/manager"
	"github.com/PaddlePaddle/FastDeploy/router/internal/router"
	scheduler_handler "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/handler"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
)

func main() {
	// Parse command line arguments
	var configPath, port string
	var splitwise bool
	flag.StringVar(&configPath, "config_path", "", "path to config file")
	flag.StringVar(&port, "port", "", "listen port of router")
	flag.BoolVar(&splitwise, "splitwise", false, "enable splitwise mode")
	flag.Parse()

	// Load configuration
	cfg, err := config.Load(configPath, port, splitwise)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize logger
	logger.Init(cfg.Log.Level, cfg.Log.Output)
	defer logger.CloseLogFile()

	// Initialize manager
	manager.Init(cfg)
	scheduler_handler.Init(cfg, manager.DefaultManager)
	registerYamlPath := cfg.Manager.RegisterPath
	manager.RegisterInstancesFromConfig(registerYamlPath)

	// Initialize router
	r := router.New(cfg)

	intervalSecs := cfg.Manager.HealthCheckIntervalSecs
	go manager.MonitorInstanceHealth(context.Background(), intervalSecs)
	intervalCleanupSecs := cfg.Scheduler.EvictionIntervalSecs
	go scheduler_handler.StartBackupCleanupTask(context.Background(), intervalCleanupSecs)

	// Start server
	addr := ":" + cfg.Server.Port
	logger.Info("Starting server on %s", addr)
	if err := r.Run(addr); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
