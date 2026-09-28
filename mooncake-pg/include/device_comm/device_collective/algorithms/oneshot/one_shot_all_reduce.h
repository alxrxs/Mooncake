#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_ALL_REDUCE_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_ALL_REDUCE_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include "device_comm/device_collective/algorithms/oneshot/one_shot_types.cuh"
#include "device_comm/device_collective/resolved_group_view.h"
#include "error_types.h"
#include "gpu_runtime.h"

namespace mooncake {

class ControlUpdateBuilder;
class DeviceCollectiveWorkspace;
class LLResources;

class OneShotAllReduceAlgorithm {
   public:
    // Workspace and protocol resources must outlive the algorithm.
    static PGResult<std::unique_ptr<OneShotAllReduceAlgorithm>> create(
        int device_index, CollectiveRuntimeBindings collective,
        const DeviceCollectiveWorkspace& workspace, const LLResources& ll);

    ~OneShotAllReduceAlgorithm() noexcept;

    OneShotAllReduceAlgorithm(const OneShotAllReduceAlgorithm&) = delete;
    OneShotAllReduceAlgorithm& operator=(const OneShotAllReduceAlgorithm&) =
        delete;

    // Construct a candidate without changing published state.
    PGResult<OneShotAllReducePlan> buildPlan(
        const ResolvedGroupView& view) const;
    PGResult<void> appendPlanUpdate(ControlUpdateBuilder& builder,
                                    const OneShotAllReducePlan& plan) const;
    PGResult<void> enqueue(const AllReduceRequest& request,
                           cudaStream_t stream) const;

   private:
    OneShotAllReduceAlgorithm(int device_index,
                              const DeviceCollectiveWorkspace& workspace,
                              const LLResources& ll) noexcept
        : workspace_(workspace), ll_(ll), device_index_(device_index) {}

    const DeviceCollectiveWorkspace& workspace_;
    const LLResources& ll_;
    int device_index_ = -1;
    OneShotAllReduceDeviceState* state_ = nullptr;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_ALL_REDUCE_H
