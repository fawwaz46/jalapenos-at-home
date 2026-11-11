// Kernels for the GPU-resident forward pass. Compiled together with
// matmul.metal, which provides the projection kernels and unpack_mxfp4.

constant uint HEAD = 128;

inline float bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }

// Sums one value per thread across a threadgroup of up to 1024 threads.
inline float group_sum(float value, threadgroup float *partial, uint lane, uint simd, uint threads) {
    value = simd_sum(value);
    if (lane == 0) partial[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < (threads + 31) / 32; ++i) total += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

inline float group_max(float value, threadgroup float *partial, uint lane, uint simd, uint threads) {
    value = simd_max(value);
    if (lane == 0) partial[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = -INFINITY;
    for (uint i = 0; i < (threads + 31) / 32; ++i) best = max(best, partial[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return best;
}

struct Rows { uint rows; uint width; };

// One token's row of the embedding table, as float32.
kernel void embed_bf16(
    device const ushort *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant Rows &d [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= d.width || gid.y >= d.rows) return;
    out[gid.y * d.width + gid.x] = bf16(table[ids[gid.y] * d.width + gid.x]);
}

// The CPU path rounds quantized embeddings to bf16, so this does too.
kernel void embed_mxfp4(
    device const uchar *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant Rows &d [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= d.width || gid.y >= d.rows) return;
    float value = unpack_mxfp4(table, scales, ids[gid.y] * d.width + gid.x);
    out[gid.y * d.width + gid.x] = as_type<float>(as_type<uint>(value) & 0xffff0000u);
}

// One threadgroup per row. Safe to run in place: every read of the row
// finishes before the barrier inside group_sum, and writes come after.
kernel void rms_norm(
    device const float *x [[buffer(0)]],
    device const ushort *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant Rows &d [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    device const float *in = x + row * d.width;
    float sum = 0.0f;
    for (uint i = tid; i < d.width; i += threads) sum += in[i] * in[i];
    float size = sqrt(group_sum(sum, partial, lane, simd, threads) / float(d.width) + 1e-6f);
    for (uint i = tid; i < d.width; i += threads) {
        out[row * d.width + i] = in[i] / size * bf16(w[i]);
    }
}

struct RopeDims { uint rows; uint heads; uint offset; };

// Rotates each (i, i + 64) pair of every head by the token's position.
kernel void rope(
    device float *x [[buffer(0)]],
    constant RopeDims &d [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x, head = gid.y, row = gid.z;
    if (i >= HEAD / 2 || head >= d.heads || row >= d.rows) return;
    float freq = 1.0f / precise::pow(1000000.0f, float(i) / float(HEAD / 2));
    float angle = float(d.offset + row) * freq;
    float c = precise::cos(angle), s = precise::sin(angle);
    uint base = (row * d.heads + head) * HEAD;
    float a = x[base + i], b = x[base + i + HEAD / 2];
    x[base + i] = a * c - b * s;
    x[base + i + HEAD / 2] = b * c + a * s;
}

struct AttentionDims { uint rows; uint q_heads; uint kv_heads; uint offset; uint stride; };

// One threadgroup of 128 threads per (query head, token). Scores go in a
// device buffer since a long context won't fit in threadgroup memory.
kernel void attention(
    device const float *q [[buffer(0)]],
    device const float *keys [[buffer(1)]],
    device const float *values [[buffer(2)]],
    device float *out [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant AttentionDims &d [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float query[HEAD];
    threadgroup float partial[32];
    uint head = group.x, row = group.y;
    uint kv_head = head / (d.q_heads / d.kv_heads);
    uint q_len = d.q_heads * HEAD, kv_len = d.kv_heads * HEAD;
    uint visible = d.offset + row + 1;
    device float *score = scores + (row * d.q_heads + head) * d.stride;

    query[tid] = q[row * q_len + head * HEAD + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = sqrt(float(HEAD));
    float best = -INFINITY;
    for (uint s = tid; s < visible; s += HEAD) {
        device const float *key = keys + s * kv_len + kv_head * HEAD;
        float dot = 0.0f;
        for (uint i = 0; i < HEAD; ++i) dot += query[i] * key[i];
        score[s] = dot / scale;
        best = max(best, score[s]);
    }
    best = group_max(best, partial, lane, simd, HEAD);

    float total = 0.0f;
    for (uint s = tid; s < visible; s += HEAD) {
        float e = precise::exp(score[s] - best);
        score[s] = e;
        total += e;
    }
    total = group_sum(total, partial, lane, simd, HEAD);
    threadgroup_barrier(mem_flags::mem_device);

    float mixed = 0.0f;
    for (uint s = 0; s < visible; ++s) {
        mixed += score[s] / total * values[s * kv_len + kv_head * HEAD + tid];
    }
    out[row * q_len + head * HEAD + tid] = mixed;
}

struct Count { uint count; };

kernel void silu_mul(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant Count &d [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= d.count) return;
    float g = gate[i];
    gate[i] = g / (1.0f + precise::exp(-g)) * up[i];
}

kernel void add(
    device float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    constant Count &d [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= d.count) return;
    x[i] += y[i];
}

// Index of the largest value in each row; ties go to the lowest index,
// matching the CPU's first-maximum argmax.
kernel void argmax_rows(
    device const float *x [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant Rows &d [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float best_value[32];
    threadgroup uint best_index[32];
    float value = -INFINITY;
    uint index = 0xffffffffu;
    for (uint i = tid; i < d.width; i += threads) {
        float v = x[row * d.width + i];
        if (v > value) { value = v; index = i; }
    }
    for (ushort step = 16; step > 0; step /= 2) {
        float other = simd_shuffle_down(value, step);
        uint other_index = simd_shuffle_down(index, step);
        if (other > value || (other == value && other_index < index)) { value = other; index = other_index; }
    }
    if (lane == 0) { best_value[simd] = value; best_index[simd] = index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint i = 1; i < (threads + 31) / 32; ++i) {
            if (best_value[i] > value || (best_value[i] == value && best_index[i] < index)) {
                value = best_value[i];
                index = best_index[i];
            }
        }
        out[row] = index;
    }
}
