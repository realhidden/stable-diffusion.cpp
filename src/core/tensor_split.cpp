#include "core/tensor_split.h"

#include <cstring>
#include <numeric>
#include <regex>

#include "ggml.h"

namespace {

    // `name` is "<block>.<marker>...", the sibling is "<block>.<suffix>".
    const ggml_tensor* sibling(const SDTensorSplitRegistry& registry,
                               const std::string& name,
                               const char* marker,
                               const std::string& suffix) {
        const size_t pos = name.rfind(marker);
        if (pos == std::string::npos) {
            return nullptr;
        }
        auto it = registry.tensors.find(name.substr(0, pos + 1) + suffix);
        return it == registry.tensors.end() ? nullptr : it->second;
    }

    int64_t lcm_with_block(const ggml_tensor* axis_0_tensor, int64_t granularity) {
        // The input-split side is sliced along its quantized rows, so every slice
        // must also be a whole number of quantization blocks.
        return axis_0_tensor == nullptr ? granularity : std::lcm(granularity, (int64_t)ggml_blck_size(axis_0_tensor->type));
    }

    ggml_backend_meta_split_state mirrored() {
        ggml_backend_meta_split_state state;
        memset(&state, 0, sizeof(state));
        state.axis       = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        state.nr[0]      = 1;
        state.n_segments = 1;
        return state;
    }

    ggml_backend_meta_split_state sliced(ggml_backend_meta_split_axis axis, int64_t ne, int64_t granularity, size_t n_devices) {
        ggml_backend_meta_split_state state;
        memset(&state, 0, sizeof(state));
        state.axis       = axis;
        state.nr[0]      = 1;
        state.n_segments = 1;
        int64_t low      = 0;
        for (size_t j = 0; j + 1 < n_devices; j++) {
            int64_t high = ne * (int64_t)(j + 1) / (int64_t)n_devices;
            high -= high % granularity;
            state.ne[j] = high - low;
            low         = high;
        }
        state.ne[n_devices - 1] = ne - low;
        return state;
    }

}  // namespace

ggml_backend_meta_split_state sd_tensor_split_state(const ggml_tensor* tensor, void* userdata) {
    const auto& registry = *static_cast<const SDTensorSplitRegistry*>(userdata);
    const std::string name(tensor->name);

    static const std::regex attn_in_weight("\\.attn\\.to_(q|k|v)\\.weight$");
    static const std::regex attn_in_bias("\\.attn\\.to_(q|k|v)\\.bias$");
    static const std::regex attn_out_weight("\\.attn\\.to_out\\.0\\.weight$");
    static const std::regex mlp_in_weight("\\.img_mlp\\.(proj|gate_layer)\\.weight$");
    static const std::regex mlp_in_bias("\\.img_mlp\\.(proj|gate_layer)\\.bias$");
    static const std::regex mlp_out_weight("\\.img_mlp\\.out\\.weight$");

    if (registry.n_devices < 2) {
        return mirrored();
    }

    const bool is_attn_in  = std::regex_search(name, attn_in_weight) || std::regex_search(name, attn_in_bias);
    const bool is_attn_out = std::regex_search(name, attn_out_weight);
    if (is_attn_in || is_attn_out) {
        // Whole heads per device: the head split of Q, K and V must match, and
        // so must the input split of the output projection they feed.
        const ggml_tensor* norm_q = sibling(registry, name, ".attn.", "attn.norm_q.weight");
        const ggml_tensor* out    = sibling(registry, name, ".attn.", "attn.to_out.0.weight");
        if (norm_q == nullptr || out == nullptr) {
            return mirrored();
        }
        const int64_t granularity = lcm_with_block(out, norm_q->ne[0]);
        const int64_t width       = out->ne[0];
        if (width % granularity != 0) {
            return mirrored();
        }
        if (is_attn_out) {
            return sliced(GGML_BACKEND_SPLIT_AXIS_0, width, granularity, registry.n_devices);
        }
        const bool bias = std::regex_search(name, attn_in_bias);
        return sliced(bias ? GGML_BACKEND_SPLIT_AXIS_0 : GGML_BACKEND_SPLIT_AXIS_1, width, granularity, registry.n_devices);
    }

    const bool is_mlp_in  = std::regex_search(name, mlp_in_weight) || std::regex_search(name, mlp_in_bias);
    const bool is_mlp_out = std::regex_search(name, mlp_out_weight);
    if (is_mlp_in || is_mlp_out) {
        const ggml_tensor* out = sibling(registry, name, ".img_mlp.", "img_mlp.out.weight");
        if (out == nullptr) {
            return mirrored();
        }
        const int64_t granularity = lcm_with_block(out, 128);
        const int64_t width       = out->ne[0];
        if (width % granularity != 0) {
            return mirrored();
        }
        if (is_mlp_out) {
            return sliced(GGML_BACKEND_SPLIT_AXIS_0, width, granularity, registry.n_devices);
        }
        const bool bias = std::regex_search(name, mlp_in_bias);
        return sliced(bias ? GGML_BACKEND_SPLIT_AXIS_0 : GGML_BACKEND_SPLIT_AXIS_1, width, granularity, registry.n_devices);
    }

    return mirrored();
}
