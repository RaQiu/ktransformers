// This file is included inside AMX_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

  bool forward_prefill_streaming_slots(int qlen, int k, const int64_t* expert_ids, const float* weights,
                                       const void* input, void* output, int64_t schedule_key) {
#ifndef _WIN32
    if (!resident_io_enabled() || !slot_pool_enabled() || qlen <= 1) return false;

    const bool stream_trace = mesh_prefill_stream_trace_enabled();
    const auto forward_start = std::chrono::steady_clock::now();
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = 0;
    std::fill(this->m_local_num_.begin(), this->m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        const int expert_id = static_cast<int>(expert_ids[i * k + j]);
        if (config_.should_skip_expert(expert_id)) {
          continue;
        }
        this->m_local_pos_[i][j] = this->m_local_num_[expert_id]++;
      }
    }

    std::vector<int> active_experts;
    active_experts.reserve(config_.expert_num);
    for (int expert_id = 0; expert_id < config_.expert_num; expert_id++) {
      if (this->m_local_num_[expert_id] > 0) {
        this->m_expert_id_map_[activated_expert] = expert_id;
        active_experts.push_back(expert_id);
        activated_expert++;
      }
    }

    if (activated_expert == 0) {
      std::memset(output, 0, sizeof(float) * static_cast<size_t>(qlen) * config_.hidden_size);
      return true;
    }
    record_prefill_frequency_counts();
    maybe_dump_prefill_expert_frequency(qlen, k, expert_ids, active_experts);

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = this->gate_up_ba_pool_;
    void* gate_bc_pool_ptr = this->gate_bc_pool_;
    void* up_bc_pool_ptr = this->up_bc_pool_;
    void* down_ba_pool_ptr = this->down_ba_pool_;
    void* down_bc_pool_ptr = this->down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0,
           used_pool_bytes_ba_down = 0, used_pool_bytes_bc_down = 0;

    for (int expert_id = 0; expert_id < config_.expert_num; expert_id++) {
      this->m_local_input_ptr_[expert_id] = this->m_local_input_ + offset * config_.hidden_size;
      this->m_local_gate_output_ptr_[expert_id] = this->m_local_gate_output_ + offset * config_.intermediate_size;
      this->m_local_up_output_ptr_[expert_id] = this->m_local_up_output_ + offset * config_.intermediate_size;
      this->m_local_down_output_ptr_[expert_id] = this->m_local_down_output_ + offset * config_.hidden_size;
      offset += this->m_local_num_[expert_id];

      if (this->m_local_num_[expert_id] == 0) {
        continue;
      }

      size_t max_m = (this->m_local_num_[expert_id] + M_STEP - 1) / M_STEP * M_STEP;
      this->gate_up_ba_[expert_id]->max_m = max_m;
      this->gate_up_ba_[expert_id]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(this->buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);

      this->gate_bc_[expert_id]->max_m = max_m;
      this->gate_bc_[expert_id]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(this->buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      this->up_bc_[expert_id]->max_m = max_m;
      this->up_bc_[expert_id]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(this->buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      this->down_ba_[expert_id]->max_m = max_m;
      this->down_ba_[expert_id]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(this->buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      this->down_bc_[expert_id]->max_m = max_m;
      this->down_bc_[expert_id]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(this->buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_a += ba_size;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }

    assert(used_pool_m <= this->pool_count_);
    assert(used_pool_bytes_a <= this->gate_up_ba_pool_bytes_);
    assert(used_pool_bytes_bc_gate <= this->gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= this->up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= this->down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= this->down_bc_pool_bytes_);

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    const auto copy_input_start = std::chrono::steady_clock::now();
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        const int expert_id = static_cast<int>(expert_ids[i * k + j]);
        if (config_.should_skip_expert(expert_id)) {
          continue;
        }
        std::memcpy(this->m_local_input_ptr_[expert_id] + this->m_local_pos_[i][j] * config_.hidden_size,
                    (ggml_bf16_t*)input + i * config_.hidden_size,
                    sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });
    const auto copy_input_end = std::chrono::steady_clock::now();

    std::vector<int> static_active_experts = active_experts;
    std::vector<int> scratch_active_experts;
    // The qlen==1 path can submit async prefetches before a later qlen>1
    // prefill chunk enters this streaming path. Drain them before the fixed
    // work-slot scheduler starts replacing bindings.
    wait_for_all_pending_prefetches(false);
    // Prefill now uses only the fixed work-slot range. Anything outside that
    // range belongs to an old scratch/global-pool policy and must be recycled
    // before current active experts are scheduled.
    if (!release_prefill_scratch_slots()) {
      std::ostringstream oss;
      oss << "AMX prefill scratch slots are still busy before scheduling layer=" << config_.layer_idx
          << " tp=" << tp_part_idx << " static_slots=" << prefill_static_slot_count_
          << " capacity=" << cache_capacity_ << resident_debug_summary();
      throw std::runtime_error(oss.str());
    }

    std::vector<uint8_t> protected_mask = mesh::build_protected_mask(config_, active_experts);
    eviction_protected_mask_ = &protected_mask;
    prefill_streaming_active_ = true;
    struct PrefillEvictionGuard {
      AMX_MOE_TP* owner;
      ~PrefillEvictionGuard() {
        owner->eviction_protected_mask_ = nullptr;
        owner->prefill_streaming_active_ = false;
      }
    } prefill_guard{this};

    bool include_mins = false;
    if constexpr (requires { gate_bb_[0]->mins; }) {
      include_mins = true;
    }

    const mesh::PrefillScratchStateCounts scratch_state_before = mesh::count_prefill_scratch_states(
        config_, tp_part_idx, scratch_active_experts, expert_states_.get(), include_mins);

    const uint64_t read_reqs_before =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
            : 0;
    const uint64_t read_bytes_before =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
            : 0;
    const int resident_before = resident_expert_count_.load(std::memory_order_acquire);

    const int work_slot_capacity = prefill_static_slot_limit();
    if (!active_experts.empty() && work_slot_capacity <= 0) {
      std::ostringstream oss;
      oss << "AMX prefill work slots have no capacity layer=" << config_.layer_idx
          << " tp=" << tp_part_idx
          << " static_slots=" << prefill_static_slot_count_
          << " active=" << active_experts.size()
          << " capacity=" << cache_capacity_ << resident_debug_summary();
      throw std::runtime_error(oss.str());
    }
    uint64_t scratch_submit_us = 0;

    ensure_prefill_full_layer_static_metadata();
    const bool full_layer_preload = prefill_full_layer_preload_active();

    auto compute_expert_group = [&](const std::vector<int>& group) -> uint64_t {
      if (group.empty()) return 0;
      ExpertReadScope expert_read_scope(this, group.size());
      for (int i = 0; i < static_cast<int>(group.size()); ++i) {
        const int expert_id = group[i];
        this->m_expert_id_map_[i] = expert_id;
        note_expert_access(expert_id);
        ensure_expert_ready(expert_id, false);
        expert_read_scope.add_expert(expert_id);
      }

      const auto compute_start = std::chrono::steady_clock::now();
      direct_or_pool(static_cast<int>(group.size()), [this](int task_id) {
        int expert_idx = this->m_expert_id_map_[task_id];
        this->gate_up_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx],
                                                this->m_local_input_ptr_[expert_idx], 0, 1);
      });

      int nth = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth * static_cast<int>(group.size()) * 2, [](int _) { T::config(); },
          [this, nth, qlen](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int expert_idx = this->m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            this->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
            if (do_up) {
              this->up_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                               this->m_local_up_output_ptr_[expert_idx], ith, nth);
            } else {
              this->gate_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                                 this->m_local_gate_output_ptr_[expert_idx], ith, nth);
            }
          },
          nullptr);

      this->apply_activation(static_cast<int>(group.size()), nth, qlen);

      pool->do_work_stealing_job(
          static_cast<int>(group.size()), nullptr,
          [this](int task_id) {
            int expert_idx = this->m_expert_id_map_[task_id];
            this->down_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx],
                                                 this->m_local_gate_output_ptr_[expert_idx], 0, 1);
          },
          nullptr);

      nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * static_cast<int>(group.size()), [](int _) { T::config(); },
          [this, nth, qlen](int task_id) {
            int expert_idx = this->m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            this->do_down_gemm(expert_idx, ith, nth, qlen);
            this->down_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                               this->m_local_down_output_ptr_[expert_idx], ith, nth);
          },
          nullptr);
      const auto compute_end = std::chrono::steady_clock::now();
      return elapsed_us(compute_start, compute_end);
    };

    uint64_t static_compute_us = 0;
    uint64_t scratch_wait_us = 0;
    uint64_t scratch_compute_us = 0;
    PrefillPromotionTiming static_timing_total;
    auto add_prefill_timing = [](PrefillPromotionTiming* dst, const PrefillPromotionTiming& src) {
      dst->state_us += src.state_us;
      dst->slot_us += src.slot_us;
      dst->slot_alloc_us += src.slot_alloc_us;
      dst->request_build_us += src.request_build_us;
      dst->reader_submit_us += src.reader_submit_us;
      dst->reader_total_us += src.reader_total_us;
      dst->reader_lock_wait_us += src.reader_lock_wait_us;
      dst->reader_sqe_prep_us += src.reader_sqe_prep_us;
      dst->reader_bookkeeping_us += src.reader_bookkeeping_us;
      dst->ring_submit_us += src.ring_submit_us;
      dst->reader_stats_us += src.reader_stats_us;
      dst->request_slice_us += src.request_slice_us;
      dst->queue_wait_us += src.queue_wait_us;
      dst->finalize_us += src.finalize_us;
      dst->queue_total_us += src.queue_total_us;
      dst->promote_calls += src.promote_calls;
      dst->allocated_slots += src.allocated_slots;
      dst->reused_slots += src.reused_slots;
      dst->read_requests += src.read_requests;
      dst->ring_flushes += src.ring_flushes;
    };

    std::vector<int> work_compute_group;
    size_t work_next = 0;
    while (work_next < active_experts.size()) {
      const size_t chunk_end =
          std::min(active_experts.size(), work_next + static_cast<size_t>(work_slot_capacity));
      work_compute_group.assign(active_experts.begin() + work_next, active_experts.begin() + chunk_end);
      if (!full_layer_preload) {
        assign_prefill_work_chunk_slots(work_compute_group);
      }
      ensure_prefill_static_residents(&work_compute_group);
      add_prefill_timing(&static_timing_total, last_prefill_static_timing_);
      static_compute_us += compute_expert_group(work_compute_group);
      work_next = chunk_end;
    }
    last_prefill_static_timing_ = static_timing_total;

    const uint64_t read_reqs_after =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
            : read_reqs_before;
    const uint64_t read_bytes_after =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
            : read_bytes_before;

    if (stream_trace) {
      mesh::PrefillStaticScratchTraceData trace_data;
      trace_data.qlen = qlen;
      trace_data.schedule_key = schedule_key;
      trace_data.active = activated_expert;
      trace_data.static_active = static_active_experts.size();
      trace_data.scratch_active = scratch_active_experts.size();
      trace_data.scratch_state = scratch_state_before;
      trace_data.static_slots = prefill_static_slot_count_;
      trace_data.scratch_slots = 0;
      trace_data.scratch_submit_us = scratch_submit_us;
      trace_data.scratch_wait_us = scratch_wait_us;
      trace_data.static_compute_us = static_compute_us;
      trace_data.scratch_compute_us = scratch_compute_us;
      trace_data.read_req_delta = read_reqs_after - read_reqs_before;
      trace_data.read_bytes_delta = read_bytes_after - read_bytes_before;
      trace_data.resident_before = resident_before;
      trace_data.resident_after = resident_expert_count_.load(std::memory_order_acquire);
      trace_data.allocated_slots = allocated_slot_count();
      trace_data.scratch_timing = PrefillPromotionTiming{};
      trace_data.static_timing = last_prefill_static_timing_;
      mesh::log_prefill_static_scratch_trace(config_, tp_part_idx, trace_data);
    }

    eviction_protected_mask_ = nullptr;
    prefill_streaming_active_ = false;

    const auto merge_start = std::chrono::steady_clock::now();
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              const int expert_id = static_cast<int>(expert_ids[i * k + j]);
              if (config_.should_skip_expert(expert_id)) {
                continue;
              }
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(this->m_local_down_output_ptr_[expert_id] +
                             this->m_local_pos_[i][j] * config_.hidden_size + e),
                  &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);
    const auto merge_end = std::chrono::steady_clock::now();

    update_lookahead_heat_from_forward(qlen, k, expert_ids, weights);
    refresh_lookahead_heat();
    trim_cache_to_capacity();
    trim_cache_for_memory_pressure();
    if (tp_part_idx == 0 && config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->maybe_dump_jsonl();
    }
    if (stream_trace) {
      const auto forward_end = std::chrono::steady_clock::now();
      mesh::log_prefill_stream_summary(config_,
                                        tp_part_idx,
                                        qlen,
                                        activated_expert,
                                        static_active_experts.size(),
                                        scratch_active_experts.size(),
                                        prefill_static_slot_count_,
                                        0,
                                        elapsed_us(copy_input_start, copy_input_end),
                                        elapsed_us(merge_start, merge_end),
                                        elapsed_us(forward_start, forward_end),
                                        resident_expert_count_.load(std::memory_order_acquire),
                                        allocated_slot_count());
    }
    return true;
#else
    (void)qlen;
    (void)k;
    (void)expert_ids;
    (void)weights;
    (void)input;
    (void)output;
    return false;
#endif
  }
