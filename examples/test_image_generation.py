#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastDeploy扩散模型图像生成测试脚本

演示所有支持的模型生成图像
"""

import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 修复导入路径
from diffusion_image_generation_production import DiffusionImageGenerator


def test_all_models():
    """测试所有支持的模型"""
    
    models = ["sd15", "sdxl", "sd3", "flux"]
    prompt = "A serene lake with mountains reflected in clear water, professional photography"
    
    print("\n" + "="*70)
    print("FastDeploy Diffusion Image Generation - Model Comparison Test")
    print("="*70)
    
    results = {}
    
    for model_name in models:
        print(f"\n\n🎨 Testing {model_name.upper()} Model")
        print("-" * 70)
        
        try:
            # 初始化生成器，使用CPU设备
            generator = DiffusionImageGenerator(
                model_name=model_name,
                device="cpu",  # 在Mac上使用CPU
                use_fp16=False,  # CPU不支持FP16
            )
            
            # 生成图像
            result = generator.generate(
                prompt=prompt,
                height=512,
                width=512,
            )
            
            results[model_name] = result["performance"]
            
            print(f"\n✅ {model_name.upper()} Generation Successful!")
            print(f"Total Time: {result['performance']['total']:.2f}s")
            
        except Exception as e:
            print(f"\n❌ {model_name.upper()} Generation Failed: {e}")
            results[model_name] = None
    
    # 性能对比
    print("\n\n" + "="*70)
    print("Performance Comparison")
    print("="*70)
    
    # 创建对比表格
    print(f"\n{'Model':<10} {'Text':<8} {'Diffusion':<12} {'VAE':<8} {'Total':<8}")
    print("-" * 50)
    
    for model_name in models:
        if results[model_name]:
            perf = results[model_name]
            print(f"{model_name:<10} "
                  f"{perf['text_encoding']:<8.2f}s "
                  f"{perf['diffusion']:<12.2f}s "
                  f"{perf['vae_decoding']:<8.2f}s "
                  f"{perf['total']:<8.2f}s")


def test_batch_generation():
    """批量生成测试"""
    
    print("\n\n" + "="*70)
    print("Batch Generation Test - Flux Model")
    print("="*70)
    
    prompts = [
        "A futuristic city with flying cars and neon lights",
        "A peaceful forest with sunlight filtering through trees",
        "A modern minimalist interior design space",
    ]
    
    try:
        generator = DiffusionImageGenerator(
            model_name="flux",
            device="cpu",  # 在Mac上使用CPU
            use_fp16=False,  # CPU不支持FP16
        )
        
        print(f"\nGenerating {len(prompts)} images with Flux model...")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Generating: {prompt[:50]}...")
            
            result = generator.generate(
                prompt=prompt,
                height=512,
                width=512,
                seed=42 + i,
            )
            
            print(f"  ✅ Generated in {result['performance']['total']:.2f}s")
        
        # 性能统计
        generator.print_performance_report()
        
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")


def test_model_configurations():
    """测试不同配置"""
    
    print("\n\n" + "="*70)
    print("Configuration Test - SD3 Model with Different Settings")
    print("="*70)
    
    configs = [
        {"steps": 10, "guidance": 3.0},
        {"steps": 20, "guidance": 5.0},
        {"steps": 50, "guidance": 7.0},
    ]
    
    try:
        generator = DiffusionImageGenerator(
            model_name="sd3",
            device="cpu",  # 在Mac上使用CPU
            use_fp16=False,  # CPU不支持FP16
        )
        
        print(f"\nGenerating images with SD3 using different configurations...")
        print(f"{'Config':<20} {'Steps':<8} {'Guidance':<10} {'Total Time':<12}")
        print("-" * 50)
        
        for config in configs:
            result = generator.generate(
                prompt="A portrait of a person in professional attire",
                num_inference_steps=config["steps"],
                guidance_scale=config["guidance"],
                seed=42,
            )
            
            config_str = f"S:{config['steps']} G:{config['guidance']:.1f}"
            print(f"{config_str:<20} {config['steps']:<8} "
                  f"{config['guidance']:<10.1f} {result['performance']['total']:<12.2f}s")
        
    except Exception as e:
        print(f" Configuration test failed: {e}")


if __name__ == "__main__":
    # 运行所有测试
    test_all_models()
    test_batch_generation()
    test_model_configurations()
    
    print("\n\n" + "="*70)
    print("✨ All Tests Completed!")
    print("="*70)