use std::{env, path::PathBuf, process::Command};

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut objects = Vec::new();
    for source in ["metal", "gpu"] {
        let object = out.join(format!("{source}.o"));
        let status = Command::new("xcrun")
            .args(["clang", "-O2", "-fobjc-arc", "-c"])
            .arg(format!("src/{source}.m"))
            .arg("-o")
            .arg(&object)
            .status()
            .expect("run clang");
        assert!(status.success(), "compile Metal bridge");
        objects.push(object);
    }
    let archive = out.join("libqwen_metal.a");
    let _ = std::fs::remove_file(&archive);
    let status = Command::new("xcrun")
        .args(["ar", "rcs"])
        .arg(archive)
        .args(objects)
        .status()
        .expect("run ar");
    assert!(status.success(), "archive Metal bridge");

    println!("cargo:rerun-if-changed=src/metal.m");
    println!("cargo:rerun-if-changed=src/matmul.metal");
    println!("cargo:rerun-if-changed=src/gpu.m");
    println!("cargo:rerun-if-changed=src/gpu.metal");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=qwen_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
