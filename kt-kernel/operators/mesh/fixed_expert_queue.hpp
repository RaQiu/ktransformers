#ifndef CPUINFER_OPERATOR_MESH_FIXED_EXPERT_QUEUE_HPP
#define CPUINFER_OPERATOR_MESH_FIXED_EXPERT_QUEUE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "expert_residency.hpp"

#ifndef _WIN32

namespace mesh {

enum class FixedExpertQueueRegion : uint8_t {
  PrefillStatic = 0,
  PrefillScratch = 1,
  DecodeMain = 2,
  BackgroundWarmFill = 3,
};

enum class FixedExpertQueueWaitMode : uint8_t {
  Blocking = 0,
  Chunked = 1,
  Background = 2,
};

struct FixedExpertQueueRequest {
  FixedExpertQueueRegion region = FixedExpertQueueRegion::DecodeMain;
  ExpertLoadSource source = ExpertLoadSource::Explicit;
  int64_t schedule_key = 0;
  int slot_begin = 0;
  int slot_end = 0;
  int max_to_submit = 0;
  FixedExpertQueueWaitMode wait_mode = FixedExpertQueueWaitMode::Blocking;
};

struct FixedExpertQueueBatch {
  explicit FixedExpertQueueBatch(FixedExpertQueueRequest req = FixedExpertQueueRequest{})
      : request(req) {}

  void clear_io() {
    promotions.clear();
    read_batch.clear();
    request_ids.clear();
  }

  void reset_timing() {
    timing = PrefillPromotionTiming{};
  }

  void reset() {
    clear_io();
    reset_timing();
  }

  void reserve_experts(size_t expert_count, size_t max_requests_per_expert = 9) {
    promotions.reserve(expert_count);
    read_batch.reserve(expert_count * max_requests_per_expert);
  }

  size_t submitted_expert_count() const {
    return promotions.size();
  }

  FixedExpertQueueRequest request;
  std::vector<BatchPromotion> promotions;
  std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
  std::vector<uint64_t> request_ids;
  PrefillPromotionTiming timing;
};

inline const char* fixed_expert_queue_region_name(FixedExpertQueueRegion region) {
  switch (region) {
    case FixedExpertQueueRegion::PrefillStatic:
      return "prefill_static";
    case FixedExpertQueueRegion::PrefillScratch:
      return "prefill_scratch";
    case FixedExpertQueueRegion::DecodeMain:
      return "decode_main";
    case FixedExpertQueueRegion::BackgroundWarmFill:
      return "background_warmfill";
  }
  return "unknown";
}

inline int fixed_expert_queue_slot_count(const FixedExpertQueueRequest& request) {
  return std::max(0, request.slot_end - request.slot_begin);
}

inline size_t fixed_expert_queue_submit_limit(const FixedExpertQueueRequest& request,
                                              size_t candidate_count) {
  if (request.max_to_submit <= 0) {
    return candidate_count;
  }
  return std::min(candidate_count, static_cast<size_t>(request.max_to_submit));
}

}  // namespace mesh

#endif  // _WIN32

#endif  // CPUINFER_OPERATOR_MESH_FIXED_EXPERT_QUEUE_HPP
