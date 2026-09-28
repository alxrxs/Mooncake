#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_PRIMITIVES_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_PRIMITIVES_CUH

#include <cuda/atomic>

#include "device_comm/device_collective/protocols/ll/ll_types.cuh"
#include "device_comm/device_transfer/transfer_lane.cuh"
#include "pg_assert.h"
#include "device_comm/device_utils/device_timeout.cuh"

namespace mooncake {

// CTA-collective LL control for one invocation.
class LLControl {
   public:
    __device__ __forceinline__ LLControl(const TransferLane& lane,
                                         LLState& state, uint64_t timeout_ticks)
        : lane_(lane), state_(state), timeout_ticks_(timeout_ticks) {}

    [[nodiscard]] __device__ __forceinline__ CollectiveStepResult
    barrier(RemotePeerList remote_peers,
            cooperative_groups::thread_block block) const {
        const uint64_t sequence = state_.barrier_sequence + 1;
        PG_ASSERT(sequence != 0);
        for (const auto& peer : remote_peers) {
            const auto rank = peer.in_group_rank;
            PG_ASSERT(rank >= 0 &&
                      static_cast<uint32_t>(rank) < state_.max_group_size);
            const auto& binding = state_.peer_bindings[rank];
            PG_ASSERT(binding.global_rank == peer.global_rank);
            SignalRequest request;
            request.signal.kind = SignalAction::Kind::Set;
            request.signal.remote_offset =
                binding.signal_offset +
                uint64_t{static_cast<uint32_t>(state_.self_rank)} *
                    sizeof(uint64_t);
            request.signal.set.value = sequence;
            request.timeout_ticks = timeout_ticks_;
            if (lane_.signal(binding.global_rank, request, block).wait(block) !=
                TransferResult::Succeeded)
                return {rank};
        }
        for (const auto& peer : remote_peers) {
            const auto rank = peer.in_group_rank;
            const auto* signal = &state_.signals[rank];
            const auto result = lane_.waitSignal(
                SignalWaitRequest{signal, sequence, timeout_ticks_}, block);
            if (result.status != SignalWaitStatus::Reached) return {rank};
        }
        if (block.thread_rank() == 0) state_.barrier_sequence = sequence;
        block.sync();
        return {};
    }

   private:
    TransferLane lane_;
    LLState& state_;
    uint64_t timeout_ticks_;
};

// Per-thread LL packet operations.
class LLPacketOps {
   public:
    __device__ __forceinline__ LLPacketOps(uint64_t sequence,
                                           uint64_t timeout_ticks)
        : tag_(LLPacket::tagFor(sequence)),
          timeout_ticks_(timeout_ticks) {}

    __device__ __forceinline__ static void initialize(LLPacket* packet) {
        packet->bits = 0;
    }

    template <typename Pack>
    __device__ __forceinline__ void storePack(
        LLPacket* destination, typename Pack::Value value) const {
        static_assert(Pack::kPayloadWords == 1 || Pack::kPayloadWords == 2);
        storeWord(destination, static_cast<uint32_t>(value));
        if constexpr (Pack::kPayloadWords == 2)
            storeWord(destination + 1, static_cast<uint32_t>(value >> 32));
    }

    template <typename Pack>
    [[nodiscard]] __device__ __forceinline__ bool loadPack(
        const LLPacket* source, typename Pack::Value& value) const {
        static_assert(Pack::kPayloadWords == 1 || Pack::kPayloadWords == 2);
        uint32_t low;
        if (!loadWord(source, low)) return false;
        value = low;
        if constexpr (Pack::kPayloadWords == 2) {
            uint32_t high;
            if (!loadWord(source + 1, high)) return false;
            value |= static_cast<typename Pack::Value>(high) << 32;
        }
        return true;
    }

   private:
    __device__ __forceinline__ void storeWord(LLPacket* destination,
                                              uint32_t value) const {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> packet(
            destination->bits);
        // The data and tag are one scalar; no other memory is published here.
        packet.store((uint64_t{tag_} << 32) | value,
                     cuda::memory_order_relaxed);
    }

    [[nodiscard]] __device__ __forceinline__ bool loadWord(
        const LLPacket* source, uint32_t& value) const {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> packet(
            const_cast<uint64_t&>(source->bits));
        uint64_t observed = packet.load(cuda::memory_order_relaxed);

        // The ready case needs no clock access.
        if (static_cast<uint32_t>(observed >> 32) == tag_) {
            value = static_cast<uint32_t>(observed);
            return true;
        }

        const uint64_t start = clock64();
        do {
            observed = packet.load(cuda::memory_order_relaxed);
            if (static_cast<uint32_t>(observed >> 32) == tag_) {
                value = static_cast<uint32_t>(observed);
                return true;
            }
        } while (!deviceTimedOut(start, timeout_ticks_));
        return false;
    }

    uint32_t tag_;
    uint64_t timeout_ticks_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_LL_PRIMITIVES_CUH
