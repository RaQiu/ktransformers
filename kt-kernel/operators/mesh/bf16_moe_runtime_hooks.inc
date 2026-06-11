// This file is included inside AMX_BF16_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
#ifndef _WIN32
    std::vector<int> active_experts;
    std::vector<int> expert_counts;
    if (lazy_weight_enabled()) {
      mesh::collect_unique_cpu_experts(config_, expert_ids, qlen * k, active_experts, &expert_counts);
      complete_ready_prefetches();
      const bool trace_bf16_prefill = mesh::bf16_prefill_trace_enabled();
      auto count_states = [&](int begin, int end, int* hits, int* cold, int* inflight) {
        *hits = 0;
        *cold = 0;
        *inflight = 0;
        for (int idx = begin; idx < end; ++idx) {
          const int expert_id = active_experts[idx];
          const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
          if (state == EXPERT_CACHED || state == EXPERT_PINNED) {
            ++(*hits);
          } else if (state == EXPERT_BASELINE) {
            ++(*cold);
          } else if (state == EXPERT_PACKING || state == EXPERT_PREFETCHING || state == EXPERT_DEMOTING) {
            ++(*inflight);
          }
        }
      };
      auto read_request_count = [&]() -> uint64_t {
        return config_.cache_stats == nullptr
                   ? 0
                   : config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed);
      };
      auto read_bytes = [&]() -> uint64_t {
        return config_.cache_stats == nullptr ? 0 : config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed);
      };
      const uint64_t bytes_per_cold_expert =
          static_cast<uint64_t>(gate_weight_bytes_ + up_weight_bytes_ + full_down_weight_bytes());

      const bool enable_wave_mode = mesh::bf16_wave_resident_enabled();
      const int wave_capacity = cache_capacity_ > 0 ? cache_capacity_ : (int)active_experts.size();
      if (enable_wave_mode && qlen > 1 && wave_capacity > 0 && (int)active_experts.size() > wave_capacity) {
        auto resident_rank = [this](int expert_id) {
          const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
          if (state == EXPERT_PINNED) return 2;
          if (state == EXPERT_CACHED) return 1;
          return 0;
        };
        std::stable_sort(active_experts.begin(), active_experts.end(), [&](int lhs, int rhs) {
          const int lhs_rank = resident_rank(lhs);
          const int rhs_rank = resident_rank(rhs);
          if (lhs_rank != rhs_rank) {
            return lhs_rank > rhs_rank;
          }
          if (expert_counts[lhs] != expert_counts[rhs]) {
            return expert_counts[lhs] > expert_counts[rhs];
          }
          return lhs < rhs;
        });

        const size_t output_elems = (size_t)qlen * (size_t)config_.hidden_size;
        auto free_aligned_f32 = [](void* ptr) {
#ifdef _WIN32
          _aligned_free(ptr);
#else
          std::free(ptr);
#endif
        };
        auto alloc_aligned_f32 = [&](size_t elem_count) {
          void* ptr = nullptr;
#ifdef _WIN32
          ptr = _aligned_malloc(sizeof(float) * elem_count, 64);
          if (ptr == nullptr) {
            throw std::bad_alloc();
          }
#else
          if (posix_memalign(&ptr, 64, sizeof(float) * elem_count) != 0 || ptr == nullptr) {
            throw std::bad_alloc();
          }
#endif
          return std::unique_ptr<float, decltype(free_aligned_f32)>(static_cast<float*>(ptr), free_aligned_f32);
        };

        auto accum_output_holder = alloc_aligned_f32(output_elems);
        auto wave_output_holder = alloc_aligned_f32(output_elems);
        float* accum_output = accum_output_holder.get();
        float* wave_output = wave_output_holder.get();
        std::fill(accum_output, accum_output + output_elems, 0.0f);
        std::vector<int64_t> wave_expert_ids((size_t)qlen * (size_t)k, -1);
        std::vector<uint8_t> in_wave(config_.expert_num, 0);

        for (int wave_begin = 0; wave_begin < (int)active_experts.size(); wave_begin += wave_capacity) {
          const int wave_end = std::min<int>(wave_begin + wave_capacity, (int)active_experts.size());
          int trace_hits = 0;
          int trace_cold = 0;
          int trace_inflight = 0;
          if (trace_bf16_prefill) {
            count_states(wave_begin, wave_end, &trace_hits, &trace_cold, &trace_inflight);
          }
          const uint64_t read_req_before = trace_bf16_prefill ? read_request_count() : 0;
          const uint64_t read_bytes_before = trace_bf16_prefill ? read_bytes() : 0;
          std::fill(in_wave.begin(), in_wave.end(), 0);
          for (int idx = wave_begin; idx < wave_end; ++idx) {
            in_wave[active_experts[idx]] = 1;
          }

          std::fill(wave_expert_ids.begin(), wave_expert_ids.end(), static_cast<int64_t>(-1));
          for (int i = 0; i < qlen * k; ++i) {
            const int expert_id = (int)expert_ids[i];
            if (!config_.should_skip_expert(expert_id) && in_wave[expert_id]) {
              wave_expert_ids[i] = expert_id;
            }
          }

          auto expert_read_scope = std::make_unique<ExpertReadScope>(this, wave_end - wave_begin);
          const auto ensure_start = std::chrono::steady_clock::now();
          for (int idx = wave_begin; idx < wave_end; ++idx) {
            const int expert_id = active_experts[idx];
            note_expert_access(expert_id);
            ensure_expert_ready(expert_id, false);
            expert_read_scope->add_expert(expert_id);
          }
          const uint64_t ensure_us = mesh::runtime_elapsed_us(ensure_start, std::chrono::steady_clock::now());

          std::fill(wave_output, wave_output + output_elems, 0.0f);
          const auto forward_start = std::chrono::steady_clock::now();
          Base::forward(qlen, k, wave_expert_ids.data(), weights, input, wave_output);
          const uint64_t forward_us = mesh::runtime_elapsed_us(forward_start, std::chrono::steady_clock::now());
          if (trace_bf16_prefill) {
            const uint64_t read_req_after = read_request_count();
            const uint64_t read_bytes_after = read_bytes();
            mesh::log_bf16_prefill_trace(config_,
                                          tp_part_idx,
                                          "wave",
                                          qlen,
                                          k,
                                          active_experts.size(),
                                          static_cast<size_t>(wave_begin),
                                          static_cast<size_t>(wave_end),
                                          trace_hits,
                                          trace_cold,
                                          trace_inflight,
                                          ensure_us,
                                          forward_us,
                                          read_req_after - read_req_before,
                                          read_bytes_after - read_bytes_before,
                                          bytes_per_cold_expert * static_cast<uint64_t>(trace_cold),
                                          resident_expert_count_.load(std::memory_order_acquire),
                                          cache_capacity_);
          }
          for (size_t i = 0; i < output_elems; ++i) {
            accum_output[i] += wave_output[i];
          }

          const bool has_resident_cache_budget =
              config_.max_tier0_experts > 0 ||
              config_.max_resident_experts > 0;
          if (!has_resident_cache_budget) {
            expert_read_scope.reset();
            for (int idx = wave_begin; idx < wave_end; ++idx) {
              demote_expert(active_experts[idx]);
            }
          }
        }

        std::memcpy(output, accum_output, sizeof(float) * output_elems);
        return;
      }

      int trace_hits = 0;
      int trace_cold = 0;
      int trace_inflight = 0;
      if (trace_bf16_prefill) {
        count_states(0, static_cast<int>(active_experts.size()), &trace_hits, &trace_cold, &trace_inflight);
      }
      const uint64_t read_req_before = trace_bf16_prefill ? read_request_count() : 0;
      const uint64_t read_bytes_before = trace_bf16_prefill ? read_bytes() : 0;
      auto expert_read_scope = std::make_unique<ExpertReadScope>(this, active_experts.size());
      const auto ensure_start = std::chrono::steady_clock::now();
      for (int expert_id : active_experts) {
        note_expert_access(expert_id);
        ensure_expert_ready(expert_id, false);
        expert_read_scope->add_expert(expert_id);
      }
      const uint64_t ensure_us = mesh::runtime_elapsed_us(ensure_start, std::chrono::steady_clock::now());

      const auto forward_start = std::chrono::steady_clock::now();
      Base::forward(qlen, k, expert_ids, weights, input, output);
      const uint64_t forward_us = mesh::runtime_elapsed_us(forward_start, std::chrono::steady_clock::now());
      if (trace_bf16_prefill) {
        const uint64_t read_req_after = read_request_count();
        const uint64_t read_bytes_after = read_bytes();
        mesh::log_bf16_prefill_trace(config_,
                                      tp_part_idx,
                                      "single",
                                      qlen,
                                      k,
                                      active_experts.size(),
                                      0,
                                      active_experts.size(),
                                      trace_hits,
                                      trace_cold,
                                      trace_inflight,
                                      ensure_us,
                                      forward_us,
                                      read_req_after - read_req_before,
                                      read_bytes_after - read_bytes_before,
                                      bytes_per_cold_expert * static_cast<uint64_t>(trace_cold),
                                      resident_expert_count_.load(std::memory_order_acquire),
                                      cache_capacity_);
      }
      const bool has_resident_cache_budget =
          config_.max_tier0_experts > 0 ||
          config_.max_resident_experts > 0;
      if (!has_resident_cache_budget && !active_experts.empty()) {
        // With zero resident-cache budget, keep MESH-loaded experts only for
        // the current forward pass.
        expert_read_scope.reset();
        for (int expert_id : active_experts) {
          demote_expert(expert_id);
        }
      }
      return;
    }
#endif
    Base::forward(qlen, k, expert_ids, weights, input, output);
  }

  void promote_expert(int expert_id) {
#ifndef _WIN32
    if (!lazy_weight_enabled()) return;
    ensure_expert_ready(expert_id, true);
#else
    (void)expert_id;
#endif
  }
  void demote_expert(int expert_id) {
#ifndef _WIN32
    if (!lazy_weight_enabled()) return;
    if (expert_id < 0 || expert_id >= config_.expert_num) return;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_BASELINE) {
        return;
      }
      if (state == EXPERT_PACKING || state == EXPERT_PREFETCHING) {
        std::this_thread::yield();
        continue;
      }
      if (state != EXPERT_CACHED && state != EXPERT_PINNED) {
        return;
      }

      uint8_t expected = state;
      if (!expert_states_[expert_id].compare_exchange_strong(
              expected, EXPERT_DEMOTING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }

      while (active_readers_[expert_id].load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
      }

      free_packed_expert(expert_id);
      note_expert_demote(expert_id);
      resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return;
    }
#else
    (void)expert_id;
#endif
  }

  bool is_expert_promoted(int expert_id) const {
#ifndef _WIN32
    if (!lazy_weight_enabled()) return false;
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    return expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_PINNED;
#else
    (void)expert_id;
    return false;
#endif
  }

  int expert_state_code(int expert_id) const {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return static_cast<int>(EXPERT_BASELINE);
    if (!lazy_weight_enabled() || expert_states_ == nullptr) return static_cast<int>(EXPERT_CACHED);
    return static_cast<int>(expert_states_[expert_id].load(std::memory_order_acquire));
#else
    (void)expert_id;
    return 0;
#endif
  }

  bool is_expert_ready_resident(int expert_id) const {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    if (!lazy_weight_enabled() || expert_states_ == nullptr) return true;
    const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
    return state == EXPERT_CACHED || state == EXPERT_PINNED;
#else
    (void)expert_id;
    return false;
#endif
  }

  bool is_expert_cold_baseline(int expert_id) const {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    if (!lazy_weight_enabled() || expert_states_ == nullptr) return false;
    return expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_BASELINE;
#else
    (void)expert_id;
    return false;
#endif
  }
