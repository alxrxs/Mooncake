#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_RESOURCES_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_RESOURCES_H

#include <cstdint>
#include <memory>

#include "control_plane/control_types.h"
#include "device_comm/device_collective/resolved_group_view.h"
#include "device_comm/device_collective/protocols/ll/ll_types.cuh"
#include "device_comm/device_transfer/transfer_region.h"
#include "error_types.h"

namespace mooncake {

class ControlUpdateBuilder;
class DeviceTransferService;

// One communicator's LL signals and progress.
class LLResources {
   public:
    static PGResult<std::unique_ptr<LLResources>> create(
        DeviceTransferService& transfer_service, InGroupRank self_rank,
        uint32_t max_group_size);

    ~LLResources() noexcept;
    LLResources(const LLResources&) = delete;
    LLResources& operator=(const LLResources&) = delete;

    [[nodiscard]] uint32_t maxGroupSize() const noexcept {
        return max_group_size_;
    }
    [[nodiscard]] LLState* state() const noexcept { return state_; }
    [[nodiscard]] const LLEndpoint& localEndpoint() const noexcept {
        return endpoint_;
    }

    PGResult<LLState> bindGroupView(const ResolvedGroupView& view) const;

    PGResult<void> appendUpdate(ControlUpdateBuilder& builder,
                                const LLState& bindings) const;

   private:
    LLResources(RegionSlice signals, InGroupRank self_rank,
                uint32_t max_group_size,
                const DeviceTransferHandle* transfer_handle,
                int device_index) noexcept;

    RegionSlice signals_;
    uint32_t max_group_size_ = 0;
    LLEndpoint endpoint_;
    LLState* state_ = nullptr;
    const DeviceTransferHandle* transfer_handle_ = nullptr;
    InGroupRank self_rank_ = kInvalidInGroupRank;
    int device_index_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_RESOURCES_H
