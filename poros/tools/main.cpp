// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
* @file main.cpp
* @author tianjinjin@baidu.com
* @date Tue Mar  9 14:43:42 CST 2021
* @brief a tool to change original serialized script module to an optimized one
**/

#include <iostream>
#include <memory>
#include <sstream>
#include <sys/time.h>
#include <gflags/gflags.h>

#include "poros/compile/compile.h"
//#include "poros/compile/poros_module.h"

DEFINE_int32(batch_size,
1, "the batch size for model inference");
DEFINE_int32(repeated_num,
1000, "how many repeated test times for a single input data");
DEFINE_string(test_mode,
"poros", "which module we test this time: that are only three option: poros/original");
DEFINE_string(module_file_path,
"../model/std_pretrained_resnet50_gpu.pt", "the model file path, replace this with a real one");
DEFINE_bool(is_dynamic,
false, "the model type, used to choose input data");

void build_test_data(int batch_size,
        std::vector<std::vector<torch::jit::IValue>> &prewarm_datas, bool is_dynamic) {



    std::vector<torch::jit::IValue> inputs;

    if (is_dynamic == false) {
        inputs.push_back(at::randn({ batch_size, 3, 224, 224}, {at::kCUDA}));
        prewarm_datas.push_back(inputs);
        return;
    }
    //max
    inputs.push_back(at::randn({16, 3, 224, 224}, {at::kCUDA}));
    prewarm_datas.push_back(inputs);
    //min
    std::vector<torch::jit::IValue> inputs2;
    inputs2.push_back(at::randn({1, 3, 224, 224}, {at::kCUDA}));
    prewarm_datas.push_back(inputs2);

    //opt
    std::vector<torch::jit::IValue> inputs3;
    inputs3.push_back(at::randn({6, 3, 224, 224}, {at::kCUDA}));
    prewarm_datas.push_back(inputs3);

}

/* load a serialized script module and optimize it 
this run as a convertion tool */
int main(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
//    gflags::SetCommandLineOption("flagfile", "./conf/gflags.conf");

    torch::jit::Module mod;
    struct timeval start, end;
    float time_use;
    //////////////////////////////////////////////////////////////////
    //step1: load the origin model file
    //////////////////////////////////////////////////////////////////
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        mod = torch::jit::load(FLAGS_module_file_path);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.msg();
        return -1;
    }
    mod.eval();
    //mod.to(at::kCPU);
    mod.to(at::kCUDA);
    //////////////////////////////////////////////////////////////////
    //step2: prepare input data
    //////////////////////////////////////////////////////////////////
    // Create a vector of inputs for std-resnet50.
    std::vector<std::vector<torch::jit::IValue> > prewarm_datas;
    build_test_data(FLAGS_batch_size, prewarm_datas, FLAGS_is_dynamic);
    //mod.forward(prewarm_datas[0]);

    std::cout << "input data is ok: prewarm_datas size: " << prewarm_datas.size() << std::endl;

    //////////////////////////////////////////////////////////////////
    //step3: change mode according to given test mode and press
    //////////////////////////////////////////////////////////////////
    int warm_up_cycle = 50;
    if (FLAGS_test_mode == "poros") {
        baidu::mirana::poros::PorosOptions option;
        option.device = baidu::mirana::poros::Device::GPU;
        option.is_dynamic = FLAGS_is_dynamic;
        option.debug = true;

        auto poros_mod = baidu::mirana::poros::Compile(mod, prewarm_datas, option);
        //poros_mod->to(at::kCUDA);
        torch::jit::getProfilingMode() = true;
        torch::jit::getExecutorMode() = true;
        torch::jit::setGraphExecutorOptimize(false);

        //warm up
        for (int i = 1; i < warm_up_cycle; i++) {
            poros_mod->forward(prewarm_datas[0]);
        }
        //real press func
        gettimeofday(&start, NULL);
        for (int i = 1; i < FLAGS_repeated_num; i++) {
            auto output = poros_mod->forward(prewarm_datas[0]);
        }
        gettimeofday(&end, NULL);

    } else if (FLAGS_test_mode == "original") {

        GRAPH_DUMP("graph info:", mod.get_method("forward").graph());
        //warm up
        for (int i = 1; i < warm_up_cycle; i++) {
            mod.forward(prewarm_datas[0]);
        }
        //real press func
        gettimeofday(&start, NULL);
        for (int i = 1; i < FLAGS_repeated_num; i++) {
            auto output = mod.forward(prewarm_datas[0]);
        }
        gettimeofday(&end, NULL);
        GRAPH_DUMP("torch.jit.last_executed_optimized_graph", torch::jit::lastExecutedOptimizedGraph());
    } else {
        std::cerr << "given test module info: " << FLAGS_test_mode.c_str() << " not supported"
                  << ", only poros/original supported";
        return -1;
    }

    //////////////////////////////////////////////////////////////////
    //step4: print press result
    //////////////////////////////////////////////////////////////////
    time_use = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / (double) 1000000;
    std::cout << "press mode: " << FLAGS_test_mode.c_str()
              << ", repeted times: " << FLAGS_repeated_num
              << ", spend time: " << time_use / FLAGS_repeated_num * 1000
              << " ms/infer" << std::endl;

    std::cout << "test done QAQ\n";
}
