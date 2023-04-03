#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clashing_extern_declarations)]
#![allow(temporary_cstring_as_ptr)]
extern crate libc;
use libc::{c_char, c_void, free};
use std::ffi::CString;
use clap::{App, Arg};

pub mod fd {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[link(name = "fastdeploy")]
extern "C" {
    pub fn snprintf(s: *const libc::c_char, n: usize, format: *const libc::c_char, _: ...) -> libc::c_int;
}

fn FDBooleanToRust(b: fd::FD_C_Bool) -> bool {
	let cFalse: fd::FD_C_Bool = 0;
	if b != cFalse {
		return true;
	}
	return false;
}

fn CpuInfer(model_file: *const c_char, image_file: *const c_char) {

    unsafe {
        let  option = fd::FD_C_CreateRuntimeOptionWrapper();
        fd::FD_C_RuntimeOptionWrapperUseCpu(option);

        let  model: *mut fd::FD_C_YOLOv5Wrapper = fd::FD_C_CreateYOLOv5Wrapper(
              model_file, CString::new("").unwrap().as_ptr(), option, fd::FD_C_ModelFormat_ONNX as i32);

        if !FDBooleanToRust(fd::FD_C_YOLOv5WrapperInitialized(model)) {
           print!("Failed to initialize.\n");
           fd::FD_C_DestroyRuntimeOptionWrapper(option);
           fd::FD_C_DestroyYOLOv5Wrapper(model);
           return;
        }

        let  image  = fd::FD_C_Imread(image_file);
        let  result = fd::FD_C_CreateDetectionResult();

        if !FDBooleanToRust(fd::FD_C_YOLOv5WrapperPredict(model, image, result)) {
            print!("Failed to predict.\n");
            fd::FD_C_DestroyRuntimeOptionWrapper(option);
            fd::FD_C_DestroyYOLOv5Wrapper(model);
            fd::FD_C_DestroyMat(image);
            free(result as *mut c_void);
            return;
        }

        let vis_im = fd::FD_C_VisDetection(image, result, 0.5, 1, 0.5);
        let vis_im_path = CString::new("vis_result_yolov5.jpg").unwrap();

        fd::FD_C_Imwrite(vis_im_path.as_ptr(), vis_im);
        print!("Visualized result saved in ./vis_result_yolov5.jpg\n");

        fd::FD_C_DestroyRuntimeOptionWrapper(option);
        fd::FD_C_DestroyYOLOv5Wrapper(model);
        fd::FD_C_DestroyDetectionResult(result);
        fd::FD_C_DestroyMat(image);
        fd::FD_C_DestroyMat(vis_im);
    }
}


fn GpuInfer(model_file: *const c_char, image_file: *const c_char) {

    unsafe {
        let  option = fd::FD_C_CreateRuntimeOptionWrapper();
        fd::FD_C_RuntimeOptionWrapperUseGpu(option, 0);

        let  model: *mut fd::FD_C_YOLOv5Wrapper = fd::FD_C_CreateYOLOv5Wrapper(
             model_file, CString::new("").unwrap().as_ptr(), option, fd::FD_C_ModelFormat_ONNX as i32);

        if !FDBooleanToRust(fd::FD_C_YOLOv5WrapperInitialized(model)) {
            print!("Failed to initialize.\n");
            fd::FD_C_DestroyRuntimeOptionWrapper(option);
            fd::FD_C_DestroyYOLOv5Wrapper(model);
            return;
        }

        let  image  = fd::FD_C_Imread(image_file);
        let  result = fd::FD_C_CreateDetectionResult();

        if !FDBooleanToRust(fd::FD_C_YOLOv5WrapperPredict(model, image, result)) {
            print!("Failed to predict.\n");
            fd::FD_C_DestroyRuntimeOptionWrapper(option);
            fd::FD_C_DestroyYOLOv5Wrapper(model);
            fd::FD_C_DestroyMat(image);
            free(result as *mut c_void);
            return;
        }

        let vis_im = fd::FD_C_VisDetection(image, result, 0.5, 1, 0.5);
        let vis_im_path = CString::new("vis_result_yolov5.jpg").unwrap();

        fd::FD_C_Imwrite(vis_im_path.as_ptr(), vis_im);
        print!("Visualized result saved in ./vis_result_yolov5.jpg\n");

        fd::FD_C_DestroyRuntimeOptionWrapper(option);
        fd::FD_C_DestroyYOLOv5Wrapper(model);
        fd::FD_C_DestroyDetectionResult(result);
        fd::FD_C_DestroyMat(image);
        fd::FD_C_DestroyMat(vis_im);
       }
}


fn main(){
    let matches = App::new("infer command")
        .version("0.1")
        .about("Infer Run Options")
        .arg(Arg::with_name("model")
                .long("model")
                .help("paddle detection model to use")
                .takes_value(true)
                .required(true))
        .arg(Arg::with_name("image")
                .long("image")
                .help("image to predict")
                .takes_value(true)
                .required(true))
        .arg(Arg::with_name("device")
                 .long("device")
                 .help("The data type of run_option is int, 0: run with cpu; 1: run with gpu")
                 .takes_value(true)
                 .required(true))
        .get_matches();

    let model_file   =  matches.value_of("model").unwrap();
    let image_file  =  matches.value_of("image").unwrap();
    let device_type =  matches.value_of("device").unwrap();

    if model_file != "" && image_file != "" {
    	if device_type == "0" {
    		CpuInfer(CString::new(model_file).unwrap().as_ptr(), CString::new(image_file).unwrap().as_ptr());
    	}else if device_type == "1" {
    		GpuInfer(CString::new(model_file).unwrap().as_ptr(), CString::new(image_file).unwrap().as_ptr());
    	}
    }else{
    	print!("Usage: cargo run -- --model path/to/model --image path/to/image --device run_option \n");
    	print!("e.g cargo run -- --model yolov5s.onnx --image 000000014439.jpg --device 0 \n");
    }
}