//! The whole forward pass on the GPU, one command buffer per pass.
//! Only token IDs come back to the CPU.

use std::cell::{Cell, RefCell, RefMut};
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::ptr::NonNull;

use crate::model::KvCache;
use crate::{HEAD_DIM, Tensor, Weights};

#[repr(C)]
struct Bind {
    buffer: *mut c_void,
    offset: u64,
    index: u32,
}

unsafe extern "C" {
    fn gpu_create(source: *const c_char) -> *mut c_void;
    fn gpu_pipeline(context: *mut c_void, name: *const c_char) -> *mut c_void;
    fn gpu_buffer(context: *mut c_void, bytes: *const c_void, length: u64) -> *mut c_void;
    fn gpu_buffer_contents(buffer: *mut c_void) -> *mut c_void;
    fn gpu_release(object: *mut c_void);
    fn gpu_begin(context: *mut c_void) -> *mut c_void;
    fn gpu_dispatch(
        pass: *mut c_void,
        pipeline: *mut c_void,
        binds: *const Bind,
        bind_count: u32,
        params: *const c_void,
        params_length: u32,
        params_index: u32,
        gx: u32,
        gy: u32,
        gz: u32,
        tx: u32,
        ty: u32,
        tz: u32,
    );
    fn gpu_finish(pass: *mut c_void) -> f64;
}

const KERNELS: [&str; 12] = [
    "project",
    "project_token",
    "project_mxfp4",
    "project_token_mxfp4",
    "embed_bf16",
    "embed_mxfp4",
    "rms_norm",
    "rope",
    "attention",
    "silu_mul",
    "add",
    "argmax_rows",
];

/// A GPU buffer in memory shared with the CPU.
pub(crate) struct Buffer {
    ptr: NonNull<c_void>,
    len: usize,
}

impl Buffer {
    fn contents(&self) -> *mut u8 {
        // The bridge returns the buffer's CPU-visible base address.
        unsafe { gpu_buffer_contents(self.ptr.as_ptr()).cast() }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // The bridge returned one retained reference for this buffer.
        unsafe { gpu_release(self.ptr.as_ptr()) }
    }
}

struct Device {
    context: NonNull<c_void>,
    pipelines: HashMap<&'static str, NonNull<c_void>>,
}

impl Device {
    fn new() -> Result<Self, String> {
        let source = CString::new(concat!(
            include_str!("matmul.metal"),
            "\n",
            include_str!("gpu.metal")
        ))
        .unwrap();
        // The bridge compiles the source before returning its retained context.
        let context = NonNull::new(unsafe { gpu_create(source.as_ptr()) })
            .ok_or("could not initialize Metal")?;
        let mut device = Self {
            context,
            pipelines: HashMap::new(),
        };
        for name in KERNELS {
            let c_name = CString::new(name).unwrap();
            let pipeline =
                NonNull::new(unsafe { gpu_pipeline(context.as_ptr(), c_name.as_ptr()) })
                    .ok_or_else(|| format!("could not build Metal kernel {name}"))?;
            device.pipelines.insert(name, pipeline);
        }
        Ok(device)
    }

    fn zeroed(&self, len: usize) -> Buffer {
        let len = len.max(4);
        // A null source asks Metal for a new zero-filled buffer of this length.
        let ptr = unsafe { gpu_buffer(self.context.as_ptr(), std::ptr::null(), len as u64) };
        Buffer {
            ptr: NonNull::new(ptr).expect("could not allocate GPU memory"),
            len,
        }
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let len = std::mem::size_of_val(data);
        if len == 0 {
            return self.zeroed(4);
        }
        // Metal copies `len` bytes from the slice before returning.
        let ptr = unsafe { gpu_buffer(self.context.as_ptr(), data.as_ptr().cast(), len as u64) };
        Buffer {
            ptr: NonNull::new(ptr).expect("could not allocate GPU memory"),
            len,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // Each pipeline and the context were returned retained by the bridge.
        for pipeline in self.pipelines.values() {
            unsafe { gpu_release(pipeline.as_ptr()) }
        }
        unsafe { gpu_release(self.context.as_ptr()) }
    }
}

/// One command buffer: kernels are recorded in order and run when finished.
struct Pass<'a> {
    device: &'a Device,
    ptr: NonNull<c_void>,
}

impl<'a> Pass<'a> {
    fn new(device: &'a Device) -> Self {
        let ptr = unsafe { gpu_begin(device.context.as_ptr()) };
        Self {
            device,
            ptr: NonNull::new(ptr).expect("could not start a GPU pass"),
        }
    }

    /// `binds` are (index, buffer, byte offset); `params` go to `params_index`.
    fn dispatch(
        &self,
        kernel: &str,
        binds: &[(u32, &Buffer, usize)],
        params: &[u32],
        params_index: u32,
        groups: [usize; 3],
        threads: [usize; 3],
    ) {
        let binds: Vec<Bind> = binds
            .iter()
            .map(|&(index, buffer, offset)| {
                debug_assert!(offset < buffer.len);
                Bind {
                    buffer: buffer.ptr.as_ptr(),
                    offset: offset as u64,
                    index,
                }
            })
            .collect();
        let u = |n: usize| u32::try_from(n).expect("dispatch size overflow");
        // Every buffer outlives the pass, and params are copied by Metal.
        unsafe {
            gpu_dispatch(
                self.ptr.as_ptr(),
                self.device.pipelines[kernel].as_ptr(),
                binds.as_ptr(),
                u(binds.len()),
                params.as_ptr().cast(),
                u(std::mem::size_of_val(params)),
                params_index,
                u(groups[0]),
                u(groups[1]),
                u(groups[2]),
                u(threads[0]),
                u(threads[1]),
                u(threads[2]),
            )
        }
    }

    /// Runs the recorded pass and waits once. Returns GPU busy seconds.
    fn finish(self) -> f64 {
        let seconds = unsafe { gpu_finish(self.ptr.as_ptr()) };
        assert!(seconds >= 0.0, "GPU pass failed");
        seconds
    }
}

struct GpuTensor {
    data: Buffer,
    scales: Option<Buffer>,
    /// Output features (rows of the weight matrix).
    out: usize,
    /// Input features (row length).
    inner: usize,
}

impl GpuTensor {
    fn new(device: &Device, tensor: &Tensor) -> Self {
        let (data, scales) = match &tensor.mxfp4 {
            Some(q) => (device.upload(&q.packed), Some(device.upload(&q.scales))),
            None => (device.upload(&tensor.data), None),
        };
        Self {
            data,
            scales,
            out: tensor.shape[0],
            inner: tensor.shape[1],
        }
    }
}

struct Layer {
    input_norm: Buffer,
    post_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    o: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    down: GpuTensor,
}

/// Per-pass working memory, reused across passes and grown when needed.
struct Scratch {
    rows: usize,
    scores: usize,
    ids: Buffer,
    out_ids: Buffer,
    x: Buffer,
    xn: Buffer,
    q: Buffer,
    attn: Buffer,
    tmp: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,
    score: Buffer,
}

/// Keys and values for every layer, kept in GPU memory.
pub(crate) struct GpuCache {
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    capacity: usize,
    width: usize,
}

impl GpuCache {
    fn reserve(&mut self, device: &Device, layers: usize, tokens: usize) {
        if tokens <= self.capacity {
            return;
        }
        let capacity = tokens.max(self.capacity * 2).max(256).next_power_of_two();
        let bytes = capacity * self.width * 4;
        let used = self.capacity * self.width * 4;
        let grow = |old: &mut Vec<Buffer>| {
            let mut fresh: Vec<Buffer> = (0..layers).map(|_| device.zeroed(bytes)).collect();
            for (new, old) in fresh.iter_mut().zip(old.iter()) {
                // The previous pass has finished, so the old buffer is idle.
                unsafe { std::ptr::copy_nonoverlapping(old.contents(), new.contents(), used) };
            }
            *old = fresh;
        };
        grow(&mut self.keys);
        grow(&mut self.values);
        self.capacity = capacity;
    }
}

pub(crate) struct GpuModel {
    device: Device,
    embed: GpuTensor,
    lm_head: Option<GpuTensor>,
    final_norm: Buffer,
    layers: Vec<Layer>,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    inter: usize,
    vocab: usize,
    scratch: RefCell<Option<Scratch>>,
    /// Total GPU busy time across passes, for profiling.
    pub(crate) gpu_seconds: Cell<f64>,
}

enum Head {
    /// Stop after the layers and return the hidden state.
    Hidden,
    /// Predict the next token after the last row, or after every row.
    Tokens { all: bool },
}

enum Output {
    Hidden(Vec<f32>),
    Tokens(Vec<usize>),
}

fn ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

impl GpuModel {
    pub(crate) fn new(weights: &Weights, layer_count: usize) -> Result<Self, String> {
        let device = Device::new()?;
        let get = |name: &str| {
            weights
                .tensors
                .get(name)
                .ok_or_else(|| format!("missing tensor {name}"))
        };
        let norm = |name: &str| -> Result<Buffer, String> {
            let tensor = get(name)?;
            if tensor.mxfp4.is_some() {
                return Err(format!("expected a bf16 norm weight: {name}"));
            }
            Ok(device.upload(&tensor.data))
        };
        let matrix = |name: &str| get(name).map(|t| GpuTensor::new(&device, t));

        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let p = format!("model.layers.{layer}");
            layers.push(Layer {
                input_norm: norm(&format!("{p}.input_layernorm.weight"))?,
                post_norm: norm(&format!("{p}.post_attention_layernorm.weight"))?,
                q_norm: norm(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: norm(&format!("{p}.self_attn.k_norm.weight"))?,
                q: matrix(&format!("{p}.self_attn.q_proj.weight"))?,
                k: matrix(&format!("{p}.self_attn.k_proj.weight"))?,
                v: matrix(&format!("{p}.self_attn.v_proj.weight"))?,
                o: matrix(&format!("{p}.self_attn.o_proj.weight"))?,
                gate: matrix(&format!("{p}.mlp.gate_proj.weight"))?,
                up: matrix(&format!("{p}.mlp.up_proj.weight"))?,
                down: matrix(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let embed = matrix("model.embed_tokens.weight")?;
        let lm_head = weights
            .tensors
            .get("lm_head.weight")
            .map(|t| GpuTensor::new(&device, t));
        let final_norm = norm("model.norm.weight")?;
        let first = layers.first().ok_or("model has no layers")?;
        let hidden = embed.inner;
        let q_heads = first.q.out / HEAD_DIM;
        let kv_heads = first.k.out / HEAD_DIM;
        let inter = first.gate.out;
        let vocab = lm_head.as_ref().unwrap_or(&embed).out;
        if q_heads % kv_heads != 0 || first.q.inner != hidden {
            return Err("unexpected attention shapes".into());
        }
        Ok(Self {
            device,
            embed,
            lm_head,
            final_norm,
            layers,
            hidden,
            q_heads,
            kv_heads,
            inter,
            vocab,
            scratch: RefCell::new(None),
            gpu_seconds: Cell::new(0.0),
        })
    }

    pub(crate) fn cache(&self) -> GpuCache {
        GpuCache {
            keys: Vec::new(),
            values: Vec::new(),
            capacity: 0,
            width: self.kv_heads * HEAD_DIM,
        }
    }

    fn scratch(&self, rows: usize, scores: usize) -> RefMut<'_, Scratch> {
        let mut slot = self.scratch.borrow_mut();
        if slot
            .as_ref()
            .is_none_or(|s| s.rows < rows || s.scores < scores)
        {
            let rows = rows.max(slot.as_ref().map_or(1, |s| s.rows)).next_power_of_two();
            let scores = scores.max(slot.as_ref().map_or(1, |s| s.scores)).next_power_of_two();
            let d = &self.device;
            let f = |n: usize| d.zeroed(n * 4);
            *slot = Some(Scratch {
                rows,
                scores,
                ids: f(rows),
                out_ids: f(rows),
                x: f(rows * self.hidden),
                xn: f(rows * self.hidden),
                q: f(rows * self.q_heads * HEAD_DIM),
                attn: f(rows * self.q_heads * HEAD_DIM),
                tmp: f(rows * self.hidden),
                gate: f(rows * self.inter),
                up: f(rows * self.inter),
                logits: f(rows * self.vocab),
                score: f(scores),
            });
        }
        RefMut::map(slot, |s| s.as_mut().unwrap())
    }

    fn matmul(&self, pass: &Pass, x: (&Buffer, usize), w: &GpuTensor, y: (&Buffer, usize), rows: usize) {
        let mut binds = vec![(0, x.0, x.1), (1, &w.data, 0), (2, y.0, y.1)];
        if let Some(scales) = &w.scales {
            binds.push((4, scales, 0));
        }
        let params = [rows as u32, w.out as u32, w.inner as u32, 0];
        let quantized = w.scales.is_some();
        if rows == 1 {
            let kernel = if quantized { "project_token_mxfp4" } else { "project_token" };
            pass.dispatch(kernel, &binds, &params, 3, [ceil(w.out, 8), 1, 1], [256, 1, 1]);
        } else {
            let kernel = if quantized { "project_mxfp4" } else { "project" };
            pass.dispatch(kernel, &binds, &params, 3, [ceil(w.out, 16), ceil(rows, 16), 1], [16, 16, 1]);
        }
    }

    fn rms_norm(&self, pass: &Pass, x: (&Buffer, usize), w: &Buffer, out: (&Buffer, usize), rows: usize, width: usize) {
        let threads = width.min(256);
        pass.dispatch(
            "rms_norm",
            &[(0, x.0, x.1), (1, w, 0), (2, out.0, out.1)],
            &[rows as u32, width as u32],
            3,
            [rows, 1, 1],
            [threads, 1, 1],
        );
    }

    fn rope(&self, pass: &Pass, x: (&Buffer, usize), rows: usize, heads: usize, offset: usize) {
        pass.dispatch(
            "rope",
            &[(0, x.0, x.1)],
            &[rows as u32, heads as u32, offset as u32],
            1,
            [1, heads, rows],
            [HEAD_DIM / 2, 1, 1],
        );
    }

    fn elementwise(&self, pass: &Pass, kernel: &str, a: &Buffer, b: &Buffer, count: usize) {
        pass.dispatch(kernel, &[(0, a, 0), (1, b, 0)], &[count as u32], 2, [ceil(count, 256), 1, 1], [256, 1, 1]);
    }

    fn run(&self, tokens: &[usize], cache: &mut KvCache, layers: usize, head: Head) -> Output {
        let rows = tokens.len();
        let offset = cache.len;
        let total = offset + rows;
        let kv_width = self.kv_heads * HEAD_DIM;
        let gpu_cache = cache.gpu.as_mut().expect("this cache belongs to another backend");
        gpu_cache.reserve(&self.device, self.layers.len(), total);
        let s = self.scratch(rows, rows * self.q_heads * total);

        let ids: Vec<u32> = tokens.iter().map(|&t| u32::try_from(t).unwrap()).collect();
        // No pass is running, and the buffer holds at least `rows` u32s.
        unsafe { std::ptr::copy_nonoverlapping(ids.as_ptr(), s.ids.contents().cast(), rows) };

        let pass = Pass::new(&self.device);
        let mut embed_binds = vec![(0, &self.embed.data, 0), (1, &s.ids, 0), (2, &s.x, 0)];
        if let Some(scales) = &self.embed.scales {
            embed_binds.push((4, scales, 0));
        }
        let embed_kernel = if self.embed.scales.is_some() { "embed_mxfp4" } else { "embed_bf16" };
        pass.dispatch(
            embed_kernel,
            &embed_binds,
            &[rows as u32, self.hidden as u32],
            3,
            [ceil(self.hidden, 256), rows, 1],
            [256, 1, 1],
        );

        let kv_offset = offset * kv_width * 4;
        for (index, layer) in self.layers.iter().take(layers).enumerate() {
            let keys = &gpu_cache.keys[index];
            let values = &gpu_cache.values[index];

            self.rms_norm(&pass, (&s.x, 0), &layer.input_norm, (&s.xn, 0), rows, self.hidden);
            self.matmul(&pass, (&s.xn, 0), &layer.q, (&s.q, 0), rows);
            // New keys and values are written straight into the cache.
            self.matmul(&pass, (&s.xn, 0), &layer.k, (keys, kv_offset), rows);
            self.matmul(&pass, (&s.xn, 0), &layer.v, (values, kv_offset), rows);
            self.rms_norm(&pass, (&s.q, 0), &layer.q_norm, (&s.q, 0), rows * self.q_heads, HEAD_DIM);
            self.rms_norm(&pass, (keys, kv_offset), &layer.k_norm, (keys, kv_offset), rows * self.kv_heads, HEAD_DIM);
            self.rope(&pass, (&s.q, 0), rows, self.q_heads, offset);
            self.rope(&pass, (keys, kv_offset), rows, self.kv_heads, offset);
            pass.dispatch(
                "attention",
                &[(0, &s.q, 0), (1, keys, 0), (2, values, 0), (3, &s.attn, 0), (4, &s.score, 0)],
                &[rows as u32, self.q_heads as u32, self.kv_heads as u32, offset as u32, total as u32],
                5,
                [self.q_heads, rows, 1],
                [HEAD_DIM, 1, 1],
            );
            self.matmul(&pass, (&s.attn, 0), &layer.o, (&s.tmp, 0), rows);
            self.elementwise(&pass, "add", &s.x, &s.tmp, rows * self.hidden);

            self.rms_norm(&pass, (&s.x, 0), &layer.post_norm, (&s.xn, 0), rows, self.hidden);
            self.matmul(&pass, (&s.xn, 0), &layer.gate, (&s.gate, 0), rows);
            self.matmul(&pass, (&s.xn, 0), &layer.up, (&s.up, 0), rows);
            self.elementwise(&pass, "silu_mul", &s.gate, &s.up, rows * self.inter);
            self.matmul(&pass, (&s.gate, 0), &layer.down, (&s.tmp, 0), rows);
            self.elementwise(&pass, "add", &s.x, &s.tmp, rows * self.hidden);
        }
        cache.len = total;

        match head {
            Head::Hidden => {
                self.record(pass.finish());
                let mut out = vec![0.0f32; rows * self.hidden];
                // The pass has finished writing x.
                unsafe { std::ptr::copy_nonoverlapping(s.x.contents().cast(), out.as_mut_ptr(), out.len()) };
                Output::Hidden(out)
            }
            Head::Tokens { all } => {
                let (first, count) = if all { (0, rows) } else { (rows - 1, 1) };
                let lm_head = self.lm_head.as_ref().unwrap_or(&self.embed);
                self.rms_norm(&pass, (&s.x, first * self.hidden * 4), &self.final_norm, (&s.xn, 0), count, self.hidden);
                self.matmul(&pass, (&s.xn, 0), lm_head, (&s.logits, 0), count);
                pass.dispatch(
                    "argmax_rows",
                    &[(0, &s.logits, 0), (1, &s.out_ids, 0)],
                    &[count as u32, self.vocab as u32],
                    2,
                    [count, 1, 1],
                    [1024, 1, 1],
                );
                self.record(pass.finish());
                let mut out = vec![0u32; count];
                // The pass has finished writing the chosen IDs.
                unsafe { std::ptr::copy_nonoverlapping(s.out_ids.contents().cast(), out.as_mut_ptr(), count) };
                Output::Tokens(out.into_iter().map(|t| t as usize).collect())
            }
        }
    }

    fn record(&self, seconds: f64) {
        self.gpu_seconds.set(self.gpu_seconds.get() + seconds);
    }

    pub(crate) fn predict(&self, tokens: &[usize], cache: &mut KvCache, all: bool) -> Vec<usize> {
        match self.run(tokens, cache, self.layers.len(), Head::Tokens { all }) {
            Output::Tokens(tokens) => tokens,
            Output::Hidden(_) => unreachable!(),
        }
    }

    pub(crate) fn hidden(&self, tokens: &[usize], layers: usize, cache: &mut KvCache) -> Vec<f32> {
        match self.run(tokens, cache, layers, Head::Hidden) {
            Output::Hidden(hidden) => hidden,
            Output::Tokens(_) => unreachable!(),
        }
    }
}
