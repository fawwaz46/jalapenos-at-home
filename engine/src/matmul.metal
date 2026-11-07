#include <metal_stdlib>
using namespace metal;

kernel void project(
    device const float *x [[buffer(0)]],
    device const ushort *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    constexpr uint tile = 16;
    threadgroup float input[tile][tile];
    threadgroup float weight[tile][tile];
    uint row = group.y * tile + local.y;
    uint col = group.x * tile + local.x;
    float sum = 0.0f;

    for (uint start = 0; start < dims.z; start += tile) {
        uint input_index = start + local.x;
        uint weight_index = start + local.y;
        input[local.y][local.x] = row < dims.x && input_index < dims.z
            ? x[row * dims.z + input_index] : 0.0f;
        weight[local.y][local.x] = col < dims.y && weight_index < dims.z
            ? as_type<float>(uint(w[col * dims.z + weight_index]) << 16) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < tile; ++k) {
            sum = fma(input[local.y][k], weight[k][local.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < dims.x && col < dims.y) {
        y[row * dims.y + col] = sum;
    }
}

kernel void project_token(
    device const float *x [[buffer(0)]],
    device const ushort *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    uint col = group * 8 + simd;
    if (col >= dims.y) return;
    float sum = 0.0f;
    for (uint k = lane; k < dims.z; k += 32) {
        sum = fma(x[k], as_type<float>(uint(w[col * dims.z + k]) << 16), sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[col] = sum;
}

// Each 32-value block has one E8M0 scale. Even elements occupy the low nibble.
inline float unpack_mxfp4(device const uchar *w, device const uchar *scales, uint index) {
    constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    uint code = (w[index / 2] >> (4 * (index % 2))) & 15;
    float value = ldexp(levels[code & 7], int(scales[index / 32]) - 127);
    return (code & 8) ? -value : value;
}

kernel void project_mxfp4(
    device const float *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    constexpr uint tile = 16;
    threadgroup float input[tile][tile];
    threadgroup float weight[tile][tile];
    uint row = group.y * tile + local.y;
    uint col = group.x * tile + local.x;
    float sum = 0.0f;

    for (uint start = 0; start < dims.z; start += tile) {
        uint input_index = start + local.x;
        uint weight_index = start + local.y;
        input[local.y][local.x] = row < dims.x && input_index < dims.z
            ? x[row * dims.z + input_index] : 0.0f;
        weight[local.y][local.x] = col < dims.y && weight_index < dims.z
            ? unpack_mxfp4(w, scales, col * dims.z + weight_index) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < tile; ++k) {
            sum = fma(input[local.y][k], weight[k][local.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < dims.x && col < dims.y) {
        y[row * dims.y + col] = sum;
    }
}

kernel void project_token_mxfp4(
    device const float *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    uint col = group * 8 + simd;
    if (col >= dims.y) return;
    float sum = 0.0f;
    for (uint k = lane; k < dims.z; k += 32) {
        sum = fma(x[k], unpack_mxfp4(w, scales, col * dims.z + k), sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[col] = sum;
}
