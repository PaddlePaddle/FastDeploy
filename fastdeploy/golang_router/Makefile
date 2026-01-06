HOMEDIR := $(shell pwd)
OUTDIR  := /usr/local/bin

export GOENV = $(HOMEDIR)/go.env

GOPKGS  := $$(go list ./...| grep -vE "vendor")

# make, make all
all: prepare compile

prepare:
	git version     # 低于 2.17.1 可能不能正常工作
	go env
	go mod download || go mod download -x  # 下载 依赖

#make compile
compile: build
build:
	go build -o $(OUTDIR)/fd-router ./cmd

# make test, test your code
test: prepare
	go test -race -timeout=120s -v -cover $(GOPKGS) -coverprofile=coverage.out | tee unittest.txt

# make clean
clean:
	go clean
	rm -rf $(OUTDIR)

# avoid filename conflict and speed up build
.PHONY: all prepare compile test package clean build
