#include "zyron/transformer/feed_forward.hpp"

namespace zyron::transformer {

FeedForward::FeedForward(
    const Config& config,
    std::uint32_t seed)
    : up_(config.hidden_size, config.intermediate_size, true, seed + 1),
      down_(config.intermediate_size, config.hidden_size, true, seed + 2),
      dropout_(config.dropout, seed + 3) {
    config.validate();
}

autograd::Variable FeedForward::forward(
    const autograd::Variable& input) {
    auto output = down_.forward(autograd::gelu(up_.forward(input)));
    return dropout_.forward(output);
}

void FeedForward::set_qat(bool enabled, std::size_t bits) noexcept {
    up_.set_qat(enabled, bits);
    down_.set_qat(enabled, bits);
}

void FeedForward::train(bool mode) noexcept {
    Layer::train(mode);
    up_.train(mode);
    down_.train(mode);
    dropout_.train(mode);
}

std::vector<autograd::Variable*> FeedForward::parameters() {
    auto result = up_.parameters();
    const auto down_params = down_.parameters();
    result.insert(result.end(), down_params.begin(), down_params.end());
    return result;
}

} // namespace zyron::transformer
