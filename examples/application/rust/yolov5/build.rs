extern crate bindgen;

use std::env;
use std::path::PathBuf;
use std::fs::canonicalize;
fn main() {
    println!("cargo:rustc-link-search=./fastdeploy-linux-x64-0.0.0/lib");
    println!("cargo:rustc-link-lib=fastdeploy");
    println!("cargo:rerun-if-changed=wrapper.h");

    let headers_dir = PathBuf::from("./fastdeploy-linux-x64-0.0.0/include");
    let headers_dir_canonical = canonicalize(headers_dir).unwrap();
    let include_path = headers_dir_canonical.to_str().unwrap();

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{include_path}"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
