// A small generic bridge for the GPU-resident path: buffers that stay on the
// GPU, pipelines by name, and one command buffer per forward pass.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface GpuContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@end
@implementation GpuContext
@end

@interface GpuPass : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> encoder;
@end
@implementation GpuPass
@end

typedef struct {
    void *buffer;
    uint64_t offset;
    uint32_t index;
} GpuBind;

void *gpu_create(const char *source) {
    @autoreleasepool {
        GpuContext *context = [GpuContext new];
        context.device = MTLCreateSystemDefaultDevice();
        if (!context.device) return NULL;
        MTLCompileOptions *options = [MTLCompileOptions new];
        // Keep float math IEEE-safe so results track the CPU reference.
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
            options.fastMathEnabled = NO;
        }
        NSError *error = nil;
        context.library = [context.device newLibraryWithSource:@(source) options:options error:&error];
        if (!context.library) {
            NSLog(@"%@", error);
            return NULL;
        }
        context.queue = [context.device newCommandQueue];
        if (!context.queue) return NULL;
        return (__bridge_retained void *)context;
    }
}

void *gpu_pipeline(void *context_ptr, const char *name) {
    @autoreleasepool {
        GpuContext *context = (__bridge GpuContext *)context_ptr;
        id<MTLFunction> function = [context.library newFunctionWithName:@(name)];
        if (!function) return NULL;
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [context.device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            NSLog(@"%@", error);
            return NULL;
        }
        return (__bridge_retained void *)pipeline;
    }
}

// Shared storage: on Apple silicon the CPU and GPU see the same memory, so the
// CPU can write inputs and read results through the contents pointer.
void *gpu_buffer(void *context_ptr, const void *bytes, uint64_t length) {
    @autoreleasepool {
        GpuContext *context = (__bridge GpuContext *)context_ptr;
        id<MTLBuffer> buffer = bytes
            ? [context.device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared]
            : [context.device newBufferWithLength:length options:MTLResourceStorageModeShared];
        return buffer ? (__bridge_retained void *)buffer : NULL;
    }
}

void *gpu_buffer_contents(void *buffer) {
    return [(__bridge id<MTLBuffer>)buffer contents];
}

void gpu_release(void *object) {
    CFRelease(object);
}

void *gpu_begin(void *context_ptr) {
    @autoreleasepool {
        GpuContext *context = (__bridge GpuContext *)context_ptr;
        GpuPass *pass = [GpuPass new];
        pass.command = [context.queue commandBuffer];
        // Serial dispatch: each kernel finishes before the next one starts.
        pass.encoder = [pass.command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (!pass.command || !pass.encoder) return NULL;
        return (__bridge_retained void *)pass;
    }
}

void gpu_dispatch(void *pass_ptr, void *pipeline, const GpuBind *binds, uint32_t bind_count,
                  const void *params, uint32_t params_length, uint32_t params_index,
                  uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
    GpuPass *pass = (__bridge GpuPass *)pass_ptr;
    id<MTLComputeCommandEncoder> encoder = pass.encoder;
    [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
    for (uint32_t i = 0; i < bind_count; ++i) {
        [encoder setBuffer:(__bridge id<MTLBuffer>)binds[i].buffer
                    offset:binds[i].offset
                   atIndex:binds[i].index];
    }
    if (params_length) [encoder setBytes:params length:params_length atIndex:params_index];
    [encoder dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}

// Submits the whole pass and waits once. Returns GPU busy time in seconds,
// or -1 if the GPU reported an error.
double gpu_finish(void *pass_ptr) {
    @autoreleasepool {
        GpuPass *pass = (__bridge_transfer GpuPass *)pass_ptr;
        [pass.encoder endEncoding];
        [pass.command commit];
        [pass.command waitUntilCompleted];
        if (pass.command.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"%@", pass.command.error);
            return -1.0;
        }
        return pass.command.GPUEndTime - pass.command.GPUStartTime;
    }
}
