#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_TYPES_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_TYPES_CUH

#include "device_comm/device_collective/device_collective_types.cuh"
#include "device_comm/device_collective/protocols/ll/ll_types.cuh"

namespace mooncake {

// Tuning cap for one exchange
inline constexpr uint32_t kMaxOneShotChunkBytes = 64 * 1024;
inline constexpr uint32_t kOneShotSlots = 2;

// Shared workspace packets are laid out as [slot][sender][packet]. Two slots
// alternate between chunks, and sender is indexed by InGroupRank.
struct OneShotBufferLayout {
    // Maximum payload bytes one sender can write to a slot.
    uint64_t slot_capacity = 0;
    uint32_t max_group_size = 0;

    [[nodiscard]] __host__ __device__ static constexpr OneShotBufferLayout make(
        uint64_t buffer_bytes, uint32_t max_group_size) {
        const uint64_t slot_bytes =
            buffer_bytes / (uint64_t{kOneShotSlots} * max_group_size);
        return {LLPacket::payloadCapacity(slot_bytes), max_group_size};
    }

    // Index of an LL packet within the payload workspace.
    [[nodiscard]] __host__ __device__ constexpr uint64_t packetIndex(
        uint32_t slot, InGroupRank sender, uint64_t packet_index = 0) const {
        return (uint64_t{slot} * max_group_size +
                static_cast<uint32_t>(sender)) *
                   LLPacket::packetCount(slot_capacity) +
               packet_index;
    }

    [[nodiscard]] __host__ __device__ constexpr uint64_t bufferBytes() const {
        return uint64_t{kOneShotSlots} * max_group_size *
               LLPacket::storageBytes(slot_capacity);
    }
};

struct OneShotAllReducePlan {
    DevicePlanStatus status = DevicePlanStatus::Unavailable;
    uint64_t view_epoch = kInvalidViewEpoch;
    char* buffer_ptr = nullptr;
    OneShotBufferLayout layout;
    uint32_t chunk_bytes = 0;
    InGroupRank self_rank = kInvalidInGroupRank;
    uint32_t self_active_index = 0;
    uint32_t participant_count = 0;
    // Active-rank order with self removed, shared by exchange, startup and
    // drain.
    CollectivePeer remote_peers[kMaxNumRanks - 1] = {};

    [[nodiscard]] __device__ __forceinline__ RemotePeerList
    remotePeers() const {
        return {remote_peers,
                participant_count == 0 ? 0 : participant_count - 1};
    }

    // Map a non-local active index into remote_peers.
    [[nodiscard]] __device__ __forceinline__ const CollectivePeer&
    remoteParticipant(uint32_t active_index) const {
        return remote_peers[active_index < self_active_index
                                ? active_index
                                : active_index - 1];
    }
};

struct alignas(64) OneShotAllReduceDeviceState {
    OneShotAllReducePlan plan;
    CollectiveRuntimeBindings collective;
    // Borrowed peer bindings and progress from the communicator's LL resources.
    LLState* ll = nullptr;
};

cudaError_t launchOneShotAllReduceKernel(const AllReduceRequest& request,
                                         OneShotAllReduceDeviceState* state,
                                         cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_TYPES_CUH
