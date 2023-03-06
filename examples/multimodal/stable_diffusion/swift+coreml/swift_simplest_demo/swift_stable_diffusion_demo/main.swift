//
//  main.swift
//  SD_pipeline
//
//  Created by lishicheng03 on 2023/1/19.
//

import Foundation
import StableDiffusion
import CoreGraphics
import ImageIO


import CoreGraphics
import CoreML
import Foundation
import StableDiffusion
import UniformTypeIdentifiers

@discardableResult func writeCGImage(_ image: CGImage, to destinationURL: URL) -> Bool {
    guard let destination = CGImageDestinationCreateWithURL(destinationURL as CFURL, kUTTypePNG, 1, nil) else { return false }
    CGImageDestinationAddImage(destination, image, nil)
    return CGImageDestinationFinalize(destination)
}


let seed: Int = 42
let prompt: String = "a pig"

let resourceURL: URL = URL.init(filePath: "/Users/lishicheng03/ml-stable-diffusion/SD2-einsum/Resources")
let outURL: URL = URL.init(filePath: "/Users/lishicheng03/ml-stable-diffusion/pics/out_pic.png")

if #available(macOS 13.1, *) {
    let config = MLModelConfiguration()
    let unit : MLComputeUnits = MLComputeUnits.all
    config.computeUnits = unit
    
    let pipeline = try StableDiffusionPipeline(resourcesAt: resourceURL, configuration: config)
    try pipeline.loadResources()
    
    let startTime = CFAbsoluteTimeGetCurrent()
    let image : CGImage = try pipeline.generateImages(prompt: prompt, seed: UInt32(seed)).first as! CGImage
    let endTime = CFAbsoluteTimeGetCurrent()
    print("代码执行时长：%f 秒", (endTime - startTime))
    
    try writeCGImage(image, to: outURL)
    
    print("Mission completed!")
} else {
    
}



