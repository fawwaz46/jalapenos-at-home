use std::ffi::{c_char, c_void};
use std::ptr::NonNull;

use crate::{Tensor, TensorF32};

unsafe extern "C" {
    fn qwen_metal_create(source: *const c_char) -> *mut c_void;
    fn qwen_metal_destroy(context: *mut c_void);
    fn qwen_metal_matmul(
        context: *mut c_void,
        x: *const f32,
        w: *const c_void,
        scales: *const u8,
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
        let (weight_data, scales) = match &weights.mxfp4 {
            Some(quantized) => {
                quantized
                    .validate(&weights.shape)
                    .expect("invalid MXFP4 weight");
                (quantized.packed.as_ptr().cast(), quantized.scales.as_ptr())
            }
            None => {
                assert_eq!(weights.data.len(), cols * inner);
                (weights.data.as_ptr().cast(), std::ptr::null())
            }
        };
        let mut data = vec![0.0; rows * cols];
        // All buffers have the sizes checked above, and the call waits for the GPU.
        let ok = unsafe {
            qwen_metal_matmul(
                self.0.as_ptr(),
                input.data.as_ptr(),
                weight_data,
                scales,
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

#[cfg(test)]
mod tests {
    use super::Metal;
    use crate::{Tensor, TensorF32, matmul};

    #[test]
    #[ignore = "requires a Metal device"]
    fn packed_and_bf16_kernels_match_cpu_for_decode_and_prefill() {
        let metal = Metal::new().unwrap();
        let packed: Vec<_> = (0..21 * 32).map(|i| (i % 256) as u8).collect();
        let scales: Vec<_> = (0..21 * 2).map(|i| 122 + (i % 7) as u8).collect();
        let quantized = Tensor::quantized(vec![21, 64], packed, scales).unwrap();
        let decoded = quantized.values(0..21 * 64);
        let bf16 = Tensor::bf16(
            vec![21, 64],
            decoded.iter().map(|v| (v.to_bits() >> 16) as u16).collect(),
        );
        for rows in [1, 19] {
            let input = TensorF32 {
                shape: vec![rows, 64],
                data: (0..rows * 64)
                    .map(|i| ((i % 37) as f32 - 18.0) / 19.0)
                    .collect(),
            };
            for weights in [&quantized, &bf16] {
                let expected = matmul(&input, weights);
                let actual = metal.matmul(&input, weights);
                assert_eq!(actual.shape, expected.shape);
                for (a, b) in actual.data.iter().zip(expected.data) {
                    assert!((a - b).abs() < 1e-3 + b.abs() * 1e-5, "{a} != {b}");
                }
            }
        }
    }
}
