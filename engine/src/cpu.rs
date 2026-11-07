use crate::{Tensor, TensorF32, bf16_to_f32};

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

pub fn matmul(input: &TensorF32, weights: &Tensor) -> TensorF32 {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANSPOSE: i32 = 111;
    const TRANSPOSE: i32 = 112;

    let rows = input.shape[0];
    let inner = input.shape[1];
    let cols = weights.shape[0];
    assert_eq!(weights.shape[1], inner);
    assert_eq!(input.data.len(), rows * inner);
    assert_eq!(weights.data.len(), cols * inner);

    let mut data = vec![0.0; rows * cols];
    // Convert a few rows at a time so the vocabulary projection doesn't need a
    // second, f32 copy of the full embedding table.
    let block = 4096;
    let mut converted = Vec::with_capacity(block.min(cols) * inner);
    for (i, tile) in weights.data.chunks(block * inner).enumerate() {
        converted.clear();
        converted.extend(tile.iter().copied().map(bf16_to_f32));
        // Each call writes a disjoint block of columns with the full output row stride.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANSPOSE,
                TRANSPOSE,
                rows.try_into().unwrap(),
                (tile.len() / inner).try_into().unwrap(),
                inner.try_into().unwrap(),
                1.0,
                input.data.as_ptr(),
                inner.try_into().unwrap(),
                converted.as_ptr(),
                inner.try_into().unwrap(),
                0.0,
                data.as_mut_ptr().add(i * block),
                cols.try_into().unwrap(),
            );
        }
    }
    TensorF32 {
        shape: vec![rows, cols],
        data,
    }
}
