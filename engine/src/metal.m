#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>

@interface QwenMetal : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> project;
@property(nonatomic, strong) id<MTLComputePipelineState> projectToken;
@property(nonatomic, strong) NSCache<NSValue *, id<MTLBuffer>> *weights;
@end

@implementation QwenMetal
@end

void *qwen_metal_create(const char *source) {
    @autoreleasepool {
        QwenMetal *metal = [QwenMetal new];
        metal.device = MTLCreateSystemDefaultDevice();
        if (!metal.device) { fprintf(stderr, "No Metal device available\n"); return NULL; }
        NSError *error = nil;
        id<MTLLibrary> library = [metal.device newLibraryWithSource:@(source)
                                                          options:nil error:&error];
        if (!library) {
            NSLog(@"%@", error);
            return NULL;
        }
        metal.project = [metal.device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"project"] error:&error];
        if (!metal.project) { NSLog(@"%@", error); return NULL; }
        metal.projectToken = [metal.device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"project_token"] error:&error];
        if (!metal.projectToken) { NSLog(@"%@", error); return NULL; }
        metal.queue = [metal.device newCommandQueue];
        metal.weights = [NSCache new];
        metal.weights.totalCostLimit = 4ull * 1024 * 1024 * 1024;
        if (!metal.project || !metal.projectToken || !metal.queue) return NULL;
        return (__bridge_retained void *)metal;
    }
}

void qwen_metal_destroy(void *context) {
    CFRelease(context);
}

bool qwen_metal_matmul(void *context, const float *x, const uint16_t *w, float *y,
                       uint32_t rows, uint32_t cols, uint32_t inner) {
    @autoreleasepool {
        QwenMetal *metal = (__bridge QwenMetal *)context;
        NSValue *key = [NSValue valueWithPointer:w];
        id<MTLBuffer> weights = [metal.weights objectForKey:key];
        if (!weights) {
            weights = [metal.device newBufferWithBytes:w
                length:(size_t)cols * inner * sizeof(*w) options:MTLResourceStorageModeShared];
            if (!weights) return false;
            [metal.weights setObject:weights forKey:key cost:weights.length];
        }
        id<MTLBuffer> input = [metal.device newBufferWithBytes:x
            length:(size_t)rows * inner * sizeof(*x) options:MTLResourceStorageModeShared];
        size_t outputBytes = (size_t)rows * cols * sizeof(*y);
        id<MTLBuffer> output = [metal.device newBufferWithLength:outputBytes
            options:MTLResourceStorageModeShared];
        if (!input || !output) return false;

        id<MTLCommandBuffer> command = [metal.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return false;
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:weights offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        simd_uint3 dims = { rows, cols, inner };
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];

        if (rows == 1 && metal.projectToken.threadExecutionWidth == 32) {
            [encoder setComputePipelineState:metal.projectToken];
            [encoder dispatchThreadgroups:MTLSizeMake((cols + 7) / 8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [encoder setComputePipelineState:metal.project];
            [encoder dispatchThreadgroups:MTLSizeMake((cols + 15) / 16, (rows + 15) / 16, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"%@", command.error);
            return false;
        }
        memcpy(y, output.contents, outputBytes);
        return true;
    }
}
