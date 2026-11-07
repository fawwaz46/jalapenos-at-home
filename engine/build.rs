use std::{env, path::PathBuf, process::Command};

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("metal.o");
    let status = Command::new("xcrun")
        .args(["clang", "-O2", "-fobjc-arc", "-c", "src/metal.m", "-o"])
        .arg(&object)
        .status()
        .expect("run clang");
    assert!(status.success(), "compile Metal bridge");
    let status = Command::new("xcrun")
        .args(["ar", "rcs"])
        .arg(out.join("libqwen_metal.a"))
        .arg(object)
        .status()
        .expect("run ar");
    assert!(status.success(), "archive Metal bridge");

    println!("cargo:rerun-if-changed=src/metal.m");
    println!("cargo:rerun-if-changed=src/matmul.metal");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=qwen_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
