#include "device_comm/device_collective/algorithms/oneshot/one_shot_all_reduce.h"

#include <algorithm>

#include <glog/logging.h>

#include "device_comm/device_collective/device_control_update.h"
#include "device_comm/device_collective/device_collective_workspace.h"
#include "device_comm/device_collective/protocols/ll/ll_resources.h"

namespace mooncake {

PGResult<std::unique_ptr<OneShotAllReduceAlgorithm>>
OneShotAllReduceAlgorithm::create(int device_index,
                                  CollectiveRuntimeBindings collective,
                                  const DeviceCollectiveWorkspace& workspace,
                                  const LLResources& ll) {
    PG_VALIDATE_ARG(ll.state() && collective.transfer_handle &&
                        collective.invocation_state &&
                        collective.control_mailbox &&
                        collective.view_epoch_signals,
                    "One-shot device bindings are incomplete");
    auto algorithm = std::unique_ptr<OneShotAllReduceAlgorithm>(
        new OneShotAllReduceAlgorithm(device_index, workspace, ll));
    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index));
    PG_TRY_CUDA(cudaMalloc(reinterpret_cast<void**>(&algorithm->state_),
                           sizeof(OneShotAllReduceDeviceState)));
    const OneShotAllReduceDeviceState initial_state{
        .plan = {},
        .collective = collective,
        .ll = ll.state(),
    };
    PG_TRY_CUDA(cudaMemcpy(algorithm->state_, &initial_state,
                           sizeof(OneShotAllReduceDeviceState),
                           cudaMemcpyHostToDevice));
    return algorithm;
}

OneShotAllReduceAlgorithm::~OneShotAllReduceAlgorithm() noexcept {
    if (!state_) return;
    auto device_guard = GpuDeviceGuard::create(device_index_);
    if (!device_guard.has_value()) {
        LOG(ERROR) << "Failed to select CUDA device while releasing one-shot "
                      "state: "
                   << device_guard.error().message;
        return;
    }
    const auto result = cudaFree(state_);
    state_ = nullptr;
    if (result != cudaSuccess) {
        LOG(ERROR) << "Failed to free one-shot device state: "
                   << cudaGetErrorString(result);
    }
}

PGResult<OneShotAllReducePlan> OneShotAllReduceAlgorithm::buildPlan(
    const ResolvedGroupView& view) const {
    if (view.self_active_index < 0) return OneShotAllReducePlan{};
    const auto max_group_size = ll_.maxGroupSize();
    const auto index = static_cast<size_t>(view.self_active_index);
    PG_VALIDATE_STATE(index < view.participants.size() &&
                          view.participants.size() <= max_group_size &&
                          max_group_size <= kMaxNumRanks,
                      "One-shot participants are outside the group");
    PG_VALIDATE_STATE(workspace_.buffer().addr() && view.buffer_size != 0 &&
                          view.buffer_size <= workspace_.buffer().size(),
                      "One-shot workspace binding is invalid");
    const auto layout =
        OneShotBufferLayout::make(view.buffer_size, max_group_size);
    OneShotAllReducePlan plan{
        .status = DevicePlanStatus::Ready,
        .view_epoch = view.epoch,
        .buffer_ptr = static_cast<char*>(workspace_.buffer().addr()),
        .layout = layout,
        .chunk_bytes = static_cast<uint32_t>(
            std::min<uint64_t>(kMaxOneShotChunkBytes, layout.slot_capacity)),
        .self_rank = view.participants[index].in_group_rank,
        .self_active_index = static_cast<uint32_t>(index),
        .participant_count = static_cast<uint32_t>(view.participants.size()),
    };
    PG_VALIDATE_STATE(plan.chunk_bytes >= sizeof(uint64_t),
                      "One-shot peer workspace is too small");
    uint32_t remote_index = 0;
    for (size_t peer = 0; peer < view.participants.size(); ++peer) {
        if (peer != index)
            plan.remote_peers[remote_index++] =
                view.participants[peer].asPeer();
    }
    return plan;
}

PGResult<void> OneShotAllReduceAlgorithm::appendPlanUpdate(
    ControlUpdateBuilder& builder, const OneShotAllReducePlan& plan) const {
    return builder.copyBytes(&state_->plan, &plan, sizeof(plan));
}

PGResult<void> OneShotAllReduceAlgorithm::enqueue(
    const AllReduceRequest& request, cudaStream_t stream) const {
    PG_TRY_CUDA(launchOneShotAllReduceKernel(request, state_, stream));
    return {};
}

}  // namespace mooncake
