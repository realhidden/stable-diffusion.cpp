#ifndef __SD_CORE_TENSOR_SPLIT_H__
#define __SD_CORE_TENSOR_SPLIT_H__

#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"

// Tensor parallelism (--split-mode tensor) on ggml's meta device.
//
// The meta device runs one graph on several devices at once: every weight is
// either mirrored on all of them or sliced along one axis, and ggml derives the
// split of every intermediate tensor from those, inserting an all-reduce where
// a result is a partial sum. All this file decides is which weights are sliced.
//
// The scheme is Megatron-style, per transformer block: the Q/K/V projections
// and the first MLP projections are split by output (axis 1), so each device
// owns whole attention heads and a slice of the hidden units; the attention
// output and the MLP down projection are split by input (axis 0), so each
// device produces a partial sum of the block output. Anything not recognised
// is mirrored, which is always correct and merely not parallel.
struct SDTensorSplitRegistry {
    size_t n_devices = 0;
    std::vector<ggml_backend_dev_t> devices;
    // The params of the runner being placed, by name, for sibling lookups
    // (a projection's split must line up with the one it feeds).
    std::unordered_map<std::string, const ggml_tensor*> tensors;
};

ggml_backend_meta_split_state sd_tensor_split_state(const ggml_tensor* tensor, void* userdata);

#endif  // __SD_CORE_TENSOR_SPLIT_H__
