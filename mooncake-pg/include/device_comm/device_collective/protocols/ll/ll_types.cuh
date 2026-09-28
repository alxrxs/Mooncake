#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_TYPES_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_TYPES_CUH

#include "device_comm/device_collective/device_collective_types.cuh"

namespace mooncake {

// One 32-bit payload word and its 32-bit readiness tag share an atomic word.
struct LLPacket {
    uint64_t bits;
    static constexpr uint64_t kPayloadBytes = sizeof(uint32_t);
    static constexpr uint64_t kStorageBytes = sizeof(uint64_t);
    static constexpr uint64_t kFirstSequence = 1;

    [[nodiscard]] __host__ __device__ static constexpr uint32_t tagFor(
        uint64_t sequence) {
        return static_cast<uint32_t>(sequence);
    }

    [[nodiscard]] __host__ __device__ static constexpr uint64_t storageBytes(
        uint64_t payload_bytes) {
        return packetCount(payload_bytes) * kStorageBytes;
    }

    [[nodiscard]] __host__ __device__ static constexpr uint64_t packetCount(
        uint64_t payload_bytes) {
        return payload_bytes / kPayloadBytes +
               (payload_bytes % kPayloadBytes != 0);
    }

    [[nodiscard]] __host__ __device__ static constexpr uint64_t payloadCapacity(
        uint64_t storage_bytes) {
        return storage_bytes / kStorageBytes * kPayloadBytes;
    }
};
static_assert(sizeof(LLPacket) == sizeof(uint64_t));

// Protocol-owned bindings, indexed by stable InGroupRank.
struct LLPeer {
    GlobalRank global_rank = kInvalidGlobalRank;
    // Remote offsets are relative to the peer's DTS region base.
    uint64_t signal_offset = 0;
};

// Device-visible LL bindings for the current View.
struct LLState {
    const DeviceTransferHandle* transfer_handle = nullptr;
    InGroupRank self_rank = kInvalidInGroupRank;
    uint32_t max_group_size = 0;
    // Barrier signals: one word per in-group rank, indexed by sender.
    uint64_t* signals = nullptr;
    // Indexed by InGroupRank.
    LLPeer peer_bindings[kMaxNumRanks] = {};
    // The sequence published by the last successful barrier.
    uint64_t barrier_sequence = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_TYPES_CUH
