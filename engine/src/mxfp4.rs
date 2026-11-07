use std::ops::Range;

const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub struct Mxfp4 {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
}

impl Mxfp4 {
    pub(crate) fn validate(&self, shape: &[usize]) -> Result<(), &'static str> {
        if shape.len() != 2 || shape.contains(&0) || !shape[1].is_multiple_of(32) {
            return Err("MXFP4 requires nonempty rows with a width divisible by 32");
        }
        let elements = shape[0]
            .checked_mul(shape[1])
            .ok_or("tensor is too large")?;
        if self.packed.len() != elements / 2 || self.scales.len() != elements / 32 {
            return Err("MXFP4 data length does not match shape");
        }
        if self.scales.contains(&255) {
            return Err("MXFP4 checkpoint contains a NaN scale");
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    pub(crate) fn value(&self, index: usize) -> f32 {
        let code = (self.packed[index / 2] >> (4 * (index % 2))) & 15;
        let exponent = i32::from(self.scales[index / 32]) - 127;
        let magnitude = LEVELS[usize::from(code & 7)] * 2.0_f32.powi(exponent);
        if code & 8 == 0 { magnitude } else { -magnitude }
    }

    pub(crate) fn values(&self, range: Range<usize>) -> Vec<f32> {
        let mut values = Vec::with_capacity(range.len());
        for block in range.start / 32..range.end.div_ceil(32) {
            let exponent = i32::from(self.scales[block]) - 127;
            let scale = 2.0_f32.powi(exponent);
            let start = range.start.max(block * 32);
            let end = range.end.min((block + 1) * 32);
            for index in start..end {
                let code = (self.packed[index / 2] >> (4 * (index % 2))) & 15;
                let magnitude = LEVELS[usize::from(code & 7)] * scale;
                values.push(if code & 8 == 0 { magnitude } else { -magnitude });
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::Mxfp4;

    #[test]
    fn smallest_e8m0_scale_and_signed_zero_decode() {
        let weights = Mxfp4 {
            packed: vec![0x82; 16],
            scales: vec![0],
        };
        let decoded = weights.values(0..32);
        assert_eq!(decoded[0].to_bits(), 0x0040_0000);
        assert_eq!(decoded[1].to_bits(), (-0.0_f32).to_bits());
    }
}
