package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"runtime/debug"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/PaddlePaddle/FastDeploy/router/internal/manager"
	"github.com/PaddlePaddle/FastDeploy/router/internal/router"
	scheduler_handler "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/handler"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
)

func main() {
	// Parse command line arguments
	var (
		configPath  string
		port        string
		splitwise   bool
		showVersion bool
	)
	flag.StringVar(&configPath, "config_path", "", "path to config file")
	flag.StringVar(&port, "port", "", "listen port of router")
	flag.BoolVar(&splitwise, "splitwise", false, "enable splitwise mode")
	flag.BoolVar(&showVersion, "version", false, "print version info")
	flag.BoolVar(&showVersion, "V", false, "print version info (shorthand)")
	flag.Parse()

	if showVersion {
		printVersion()
		return
	}

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

func printVersion() {
	if info, ok := debug.ReadBuildInfo(); ok {
		var (
			commit  = "unknown"
			vcsTime = "unknown"
			dirty   = "unknown"
		)

		for _, s := range info.Settings {
			switch s.Key {
			case "vcs.revision":
				if len(s.Value) >= 7 {
					commit = s.Value[:7]
				} else {
					commit = s.Value
				}
			case "vcs.time":
				vcsTime = s.Value
			case "vcs.modified":
				dirty = s.Value
			}
		}

		fmt.Printf("golang-router\n")
		fmt.Printf("  version:   %s\n", info.Main.Version)
		fmt.Printf("  commit:    %s\n", commit)
		fmt.Printf("  dirty:     %s\n", dirty)
		fmt.Printf("  vcsTime:   %s\n", vcsTime)
		fmt.Printf("  module:    %s\n", info.Main.Path)
		return
	}

	fmt.Println("version info not available")
}
