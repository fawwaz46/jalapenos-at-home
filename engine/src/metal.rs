use std::ffi::{c_char, c_void};
use std::ptr::NonNull;

use crate::{Tensor, TensorF32};

unsafe extern "C" {
    fn qwen_metal_create(source: *const c_char) -> *mut c_void;
    fn qwen_metal_destroy(context: *mut c_void);
    fn qwen_metal_matmul(
        context: *mut c_void,
        x: *const f32,
        w: *const u16,
        y: *mut f32,
        rows: u32,
        cols: u32,
        inner: u32,
    ) -> bool;
}

pub(crate) struct Metal(NonNull<c_void>);

impl Metal {
    pub(crate) fn new() -> Result<Self, &'static str> {
        let source = concat!(include_str!("matmul.metal"), "\0");
        // The bridge compiles the source before returning its retained context.
        let context = unsafe { qwen_metal_create(source.as_ptr().cast()) };
        NonNull::new(context)
            .map(Self)
            .ok_or("could not initialize Metal")
    }

    // Only called with immutable weights owned by the same model as this context.
    pub(crate) fn matmul(&self, input: &TensorF32, weights: &Tensor) -> TensorF32 {
        let rows = input.shape[0];
        let cols = weights.shape[0];
        let inner = input.shape[1];
        assert_eq!(weights.shape[1], inner);
        assert_eq!(input.data.len(), rows * inner);
        assert_eq!(weights.data.len(), cols * inner);
        let mut data = vec![0.0; rows * cols];
        // All buffers have the sizes checked above, and the call waits for the GPU.
        let ok = unsafe {
            qwen_metal_matmul(
                self.0.as_ptr(),
                input.data.as_ptr(),
                weights.data.as_ptr(),
                data.as_mut_ptr(),
                rows.try_into().unwrap(),
                cols.try_into().unwrap(),
                inner.try_into().unwrap(),
            )
        };
        assert!(ok, "Metal projection failed");
        TensorF32 {
            shape: vec![rows, cols],
            data,
        }
    }
}

impl Drop for Metal {
    fn drop(&mut self) {
        // The bridge returned one retained reference for this context.
        unsafe { qwen_metal_destroy(self.0.as_ptr()) }
    }
}
