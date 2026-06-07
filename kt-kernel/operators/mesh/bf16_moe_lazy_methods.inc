// This file is included inside AMX_BF16_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

  void set_weight_buffers(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

#ifndef _WIN32
  bool mesh_load_weights_from_lazy_sources() {
    if (!lazy_weight_enabled()) return false;
    validate_iouring_config();
    return true;
  }

  void cleanup_lazy_weight_state() {
    if (!lazy_weight_enabled()) return;
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      if (expert_id < static_cast<int>(pending_prefetches_.size()) && pending_prefetches_[expert_id].task_id != 0) {
        clear_pending_prefetch(expert_id, true);
      }
    }
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      free_packed_expert(expert_id);
    }
    expert_load_scheduler_.clear_keep_capacity(config_.expert_num);
  }

  void initialize_lazy_weight_state() {
    const int en = config_.expert_num;
    packed_gate_owner_ = std::make_unique<void*[]>(en);
    std::fill_n(packed_gate_owner_.get(), en, nullptr);
    packed_up_owner_ = std::make_unique<void*[]>(en);
    std::fill_n(packed_up_owner_.get(), en, nullptr);
    packed_down_owner_ = std::make_unique<void*[]>(en);
    std::fill_n(packed_down_owner_.get(), en, nullptr);
    expert_states_ = std::make_unique<std::atomic<uint8_t>[]>(en);
    active_readers_ = std::make_unique<std::atomic<uint32_t>[]>(en);
    pending_prefetches_.assign(en, PendingPrefetch{});
    expert_load_scheduler_.reset(en);
    resident_expert_count_.store(0, std::memory_order_relaxed);
    active_load_count_.store(0, std::memory_order_relaxed);
    eviction_cursor_.store(0, std::memory_order_relaxed);
    gate_weight_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    up_weight_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    down_weight_bytes_ = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
    const int configured_resident = config_.max_resident_experts > 0 ? config_.max_resident_experts : config_.max_tier0_experts;
    cache_capacity_ = configured_resident <= 0
                          ? 0
                          : std::min(config_.expert_num, std::max(configured_resident, config_.num_experts_per_tok));
    resident_policy_.reset(en, config_.resident_cache_policy);
    full_intermediate_size_ =
        config_.intermediate_size * std::max(1, config_.pool != nullptr ? config_.pool->config.subpool_count : 1);

    if (config_.pool != nullptr && tp_part_idx < (int)config_.pool->config.subpool_numa_map.size()) {
      numa_node_ = config_.pool->config.subpool_numa_map[tp_part_idx];
    }

    for (int expert_id = 0; expert_id < en; ++expert_id) {
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_relaxed);
      active_readers_[expert_id].store(0, std::memory_order_relaxed);
      std::free(gate_bb_[expert_id]->b);
      std::free(up_bb_[expert_id]->b);
      std::free(down_bb_[expert_id]->b);
      gate_bb_[expert_id]->b = nullptr;
      up_bb_[expert_id]->b = nullptr;
      down_bb_[expert_id]->b = nullptr;
    }
  }

  void validate_iouring_config() const {
    if (!iouring_enabled()) return;
    mesh::validate_bf16_iouring_config(config_,
                                       tp_part_idx,
                                       cache_capacity_,
                                       gate_weight_bytes_,
                                       up_weight_bytes_,
                                       full_down_weight_bytes(),
                                       down_weight_bytes_);
  }

  void acquire_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_add(1, std::memory_order_acq_rel);
  }

  void release_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_sub(1, std::memory_order_acq_rel);
  }

  using ExpertReadScope = mesh::ExpertReadScope<AMX_BF16_MOE_TP<T>>;

  void free_packed_expert(int expert_id) {
    void* gate_owner = packed_gate_owner_[expert_id];
    void* up_owner = packed_up_owner_[expert_id];
    void* down_owner = packed_down_owner_[expert_id];
    packed_gate_owner_[expert_id] = nullptr;
    packed_up_owner_[expert_id] = nullptr;
    packed_down_owner_[expert_id] = nullptr;
    gate_bb_[expert_id]->b = nullptr;
    up_bb_[expert_id]->b = nullptr;
    down_bb_[expert_id]->b = nullptr;

    if (gate_owner) {
      numa_free(gate_owner, gate_weight_bytes_);
    }
    if (up_owner) {
      numa_free(up_owner, up_weight_bytes_);
    }
    if (down_owner) {
      numa_free(down_owner, down_weight_bytes_);
    }
  }

  void note_expert_access(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
    resident_policy_.note_access(expert_id, state == EXPERT_CACHED || state == EXPERT_PINNED);
  }

  void note_expert_insert(int expert_id, bool pinned) { resident_policy_.on_insert(expert_id, pinned); }

  void note_expert_pin(int expert_id) { resident_policy_.on_pin(expert_id); }

  void note_expert_demote(int expert_id) { resident_policy_.on_demote(expert_id); }

  int select_eviction_victim(int exclude_expert_id) {
    return resident_policy_.pick_victim(
        config_.expert_num, exclude_expert_id, static_cast<uint8_t>(EXPERT_CACHED),
        [this](int expert_id) { return expert_states_[expert_id].load(std::memory_order_acquire); },
        [this](int expert_id) { return active_readers_[expert_id].load(std::memory_order_acquire); });
  }

  bool evict_one_cached_expert(int exclude_expert_id) {
    if (config_.expert_num <= 1) return false;

    for (int attempt = 0; attempt < config_.expert_num; ++attempt) {
      const int victim = select_eviction_victim(exclude_expert_id);
      if (victim < 0) {
        return false;
      }
      if (victim == exclude_expert_id) {
        continue;
      }

      uint8_t expected = EXPERT_CACHED;
      if (!expert_states_[victim].compare_exchange_strong(
              expected, EXPERT_DEMOTING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }

      if (active_readers_[victim].load(std::memory_order_acquire) != 0) {
        expert_states_[victim].store(EXPERT_CACHED, std::memory_order_release);
        continue;
      }

      free_packed_expert(victim);
      note_expert_demote(victim);
      resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      expert_states_[victim].store(EXPERT_BASELINE, std::memory_order_release);
      mesh::record_cache_eviction_demote(config_);
      return true;
    }

    return false;
  }

  void record_iouring_read_stats(const std::vector<ktransformers::AsyncExpertReader::ReadRequest>& read_batch) {
    mesh::record_iouring_read_stats(config_.cache_stats, config_.enable_cache_stats, read_batch);
  }

  struct Bf16IouringReadPlan {
    ExpertFileSlot gate_slot;
    ExpertFileSlot up_slot;
    ExpertFileSlot down_slot;
    mesh::DirectReadSpan gate_span;
    mesh::DirectReadSpan up_span;
    mesh::DirectReadSpan down_span;
  };

  Bf16IouringReadPlan bf16_iouring_read_plan(int expert_id) const {
    Bf16IouringReadPlan plan;
    plan.gate_slot =
        mesh::file_slot_at(config_, tp_part_idx, config_.gate_file_slots, "gate.weight", expert_id, true);
    plan.up_slot =
        mesh::file_slot_at(config_, tp_part_idx, config_.up_file_slots, "up.weight", expert_id, true);
    plan.down_slot =
        mesh::file_slot_at(config_, tp_part_idx, config_.down_file_slots, "down.weight", expert_id, true);
    plan.gate_span = mesh::make_direct_read_span(config_, plan.gate_slot);
    plan.up_span = mesh::make_direct_read_span(config_, plan.up_slot);
    plan.down_span = mesh::make_direct_read_span(config_, plan.down_slot);
    return plan;
  }

  bool allocate_bf16_iouring_raw_buffers(const Bf16IouringReadPlan& plan,
                                         void** gate_io,
                                         void** up_io,
                                         void** down_full_io) const {
    if (gate_io == nullptr || up_io == nullptr || down_full_io == nullptr) return false;
    *gate_io = numa_alloc_onnode(plan.gate_span.size, numa_node_);
    *up_io = numa_alloc_onnode(plan.up_span.size, numa_node_);
    *down_full_io = numa_alloc_onnode(plan.down_span.size, numa_node_);
    if (*gate_io == nullptr || *up_io == nullptr || *down_full_io == nullptr) {
      if (*gate_io) {
        numa_free(*gate_io, plan.gate_span.size);
        *gate_io = nullptr;
      }
      if (*up_io) {
        numa_free(*up_io, plan.up_span.size);
        *up_io = nullptr;
      }
      if (*down_full_io) {
        numa_free(*down_full_io, plan.down_span.size);
        *down_full_io = nullptr;
      }
      return false;
    }
    return true;
  }

  void free_bf16_iouring_raw_buffers(const Bf16IouringReadPlan& plan,
                                     void* gate_io,
                                     void* up_io,
                                     void* down_full_io) const {
    if (gate_io != nullptr) numa_free(gate_io, plan.gate_span.size);
    if (up_io != nullptr) numa_free(up_io, plan.up_span.size);
    if (down_full_io != nullptr) numa_free(down_full_io, plan.down_span.size);
  }

  void free_bf16_iouring_raw_buffers_for_expert(int expert_id,
                                                void* gate_io,
                                                void* up_io,
                                                void* down_full_io) const {
    if (gate_io == nullptr && up_io == nullptr && down_full_io == nullptr) return;
    const Bf16IouringReadPlan plan = bf16_iouring_read_plan(expert_id);
    free_bf16_iouring_raw_buffers(plan, gate_io, up_io, down_full_io);
  }

  void append_bf16_iouring_read_requests_for_expert(
      int expert_id,
      const Bf16IouringReadPlan& plan,
      void* gate_io,
      void* up_io,
      void* down_full_io,
      ktransformers::AsyncExpertReader::ReadPriority priority,
      std::vector<ktransformers::AsyncExpertReader::ReadRequest>* read_batch) const {
    if (read_batch == nullptr) return;
    auto append_read = [&](const ExpertFileSlot& slot, const mesh::DirectReadSpan& span, void* io_buffer) {
      ktransformers::AsyncExpertReader::ReadRequest request{
          expert_id,
          slot.fd,
          io_buffer,
          span.size,
          span.offset,
          0,
          priority};
      request.min_success_size = span.min_success_size;
      read_batch->push_back(request);
    };
    append_read(plan.gate_slot, plan.gate_span, gate_io);
    append_read(plan.up_slot, plan.up_span, up_io);
    append_read(plan.down_slot, plan.down_span, down_full_io);
  }

  int cache_capacity_for_debug() const { return cache_capacity_; }

  void pack_bf16_buffer_b(typename T::BufferB* buffer, ggml_bf16_t* raw, int n) {
    const int nth = std::max(1, T::recommended_nth(n));
    for (int ith = 0; ith < nth; ++ith) {
      buffer->from_mat(raw, ith, nth);
    }
  }

  bool read_and_pack_iouring_expert(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
    const auto& gate_slot =
        mesh::file_slot_at(config_, tp_part_idx, config_.gate_file_slots, "gate.weight", expert_id, true);
    const auto& up_slot =
        mesh::file_slot_at(config_, tp_part_idx, config_.up_file_slots, "up.weight", expert_id, true);
    const auto& down_slot =
        mesh::file_slot_at(config_, tp_part_idx, config_.down_file_slots, "down.weight", expert_id, true);
    const auto gate_span = mesh::make_direct_read_span(config_, gate_slot);
    const auto up_span = mesh::make_direct_read_span(config_, up_slot);
    const auto down_span = mesh::make_direct_read_span(config_, down_slot);

    void* gate_io = numa_alloc_onnode(gate_span.size, numa_node_);
    void* up_io = numa_alloc_onnode(up_span.size, numa_node_);
    void* down_full_io = numa_alloc_onnode(down_span.size, numa_node_);
    if (gate_io == nullptr || up_io == nullptr || down_full_io == nullptr) {
      if (gate_io) numa_free(gate_io, gate_span.size);
      if (up_io) numa_free(up_io, up_span.size);
      if (down_full_io) numa_free(down_full_io, down_span.size);
      mesh::log_bf16_iouring_temp_alloc_failure(expert_id, gate_io, up_io, down_full_io);
      return false;
    }
    auto* gate_raw = reinterpret_cast<ggml_bf16_t*>(static_cast<char*>(gate_io) + gate_span.payload_shift);
    auto* up_raw = reinterpret_cast<ggml_bf16_t*>(static_cast<char*>(up_io) + up_span.payload_shift);
    auto* down_full_raw = reinterpret_cast<ggml_bf16_t*>(static_cast<char*>(down_full_io) + down_span.payload_shift);

    std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
    read_batch.reserve(3);
    auto append_read = [&](const ExpertFileSlot& slot, const mesh::DirectReadSpan& span, void* io_buffer) {
      ktransformers::AsyncExpertReader::ReadRequest request{
          expert_id,
          slot.fd,
          io_buffer,
          span.size,
          span.offset,
          0,
          ktransformers::AsyncExpertReader::ReadPriority::Demand};
      request.min_success_size = span.min_success_size;
      read_batch.push_back(request);
    };
    append_read(gate_slot, gate_span, gate_io);
    append_read(up_slot, up_span, up_io);
    append_read(down_slot, down_span, down_full_io);

    record_iouring_read_stats(read_batch);
    std::vector<uint64_t> request_ids = config_.async_reader->submit_reads(read_batch);
    const bool ok = config_.async_reader->wait_for_requests(request_ids, 30000);
    if (!ok) {
      mesh::log_bf16_iouring_read_failure(expert_id, request_ids, config_.async_reader);
      numa_free(gate_io, gate_span.size);
      numa_free(up_io, up_span.size);
      numa_free(down_full_io, down_span.size);
      return false;
    }

    gate_bb_[expert_id]->set_data(gate_owner);
    up_bb_[expert_id]->set_data(up_owner);
    down_bb_[expert_id]->set_data(down_owner);

    pack_bf16_buffer_b(gate_bb_[expert_id].get(), gate_raw, config_.intermediate_size);
    pack_bf16_buffer_b(up_bb_[expert_id].get(), up_raw, config_.intermediate_size);

    if (full_intermediate_size_ == config_.intermediate_size) {
      pack_bf16_buffer_b(down_bb_[expert_id].get(), down_full_raw, config_.hidden_size);
    } else {
      std::vector<ggml_bf16_t> down_contiguous((size_t)config_.hidden_size * (size_t)config_.intermediate_size);
      const size_t down_col_offset = (size_t)tp_part_idx * (size_t)config_.intermediate_size;
      const auto* src = down_full_raw;
      for (int row = 0; row < config_.hidden_size; ++row) {
        const ggml_bf16_t* src_row = src + (size_t)row * (size_t)full_intermediate_size_ + down_col_offset;
        ggml_bf16_t* dst_row = down_contiguous.data() + (size_t)row * (size_t)config_.intermediate_size;
        std::memcpy(dst_row, src_row, sizeof(ggml_bf16_t) * (size_t)config_.intermediate_size);
      }
      pack_bf16_buffer_b(down_bb_[expert_id].get(), down_contiguous.data(), config_.hidden_size);
    }

    numa_free(gate_io, gate_span.size);
    numa_free(up_io, up_span.size);
    numa_free(down_full_io, down_span.size);
    return true;
  }

  bool begin_external_iouring_pack(int expert_id, bool pin_after_pack) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PINNED) {
        return false;
      }
      if (state == EXPERT_CACHED) {
        if (!pin_after_pack) {
          return false;
        }
        uint8_t expected = EXPERT_CACHED;
        if (expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PINNED, std::memory_order_acq_rel, std::memory_order_acquire)) {
          note_expert_pin(expert_id);
          return false;
        }
        continue;
      }
      if (state == EXPERT_BASELINE) {
        if (cache_capacity_ > 0) {
          while (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
            if (!evict_one_cached_expert(expert_id)) {
              break;
            }
          }
          if (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
            return false;
          }
        }

        uint8_t expected = EXPERT_BASELINE;
        if (expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PACKING, std::memory_order_acq_rel, std::memory_order_acquire)) {
          return true;
        }
        continue;
      }
      if (state == EXPERT_PACKING || state == EXPERT_DEMOTING) {
        std::this_thread::yield();
        continue;
      }
      return false;
    }
  }

  void cancel_external_iouring_pack(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    uint8_t expected = EXPERT_PACKING;
    (void)expert_states_[expert_id].compare_exchange_strong(
        expected, EXPERT_BASELINE, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  bool pack_iouring_expert_from_raw_slices(int expert_id,
                                           const ggml_bf16_t* gate_raw,
                                           const ggml_bf16_t* up_raw,
                                           const ggml_bf16_t* down_full_raw,
                                           bool pin_after_pack) {
    void* gate_owner = numa_alloc_onnode(gate_weight_bytes_, numa_node_);
    void* up_owner = numa_alloc_onnode(up_weight_bytes_, numa_node_);
    void* down_owner = numa_alloc_onnode(down_weight_bytes_, numa_node_);
    if (gate_owner == nullptr || up_owner == nullptr || down_owner == nullptr) {
      mesh::log_bf16_numa_alloc_failure(expert_id,
                                        gate_weight_bytes_,
                                        up_weight_bytes_,
                                        down_weight_bytes_,
                                        gate_owner,
                                        up_owner,
                                        down_owner,
                                        numa_node_);
      if (gate_owner) numa_free(gate_owner, gate_weight_bytes_);
      if (up_owner) numa_free(up_owner, up_weight_bytes_);
      if (down_owner) numa_free(down_owner, down_weight_bytes_);
      cancel_external_iouring_pack(expert_id);
      return false;
    }

    gate_bb_[expert_id]->set_data(gate_owner);
    up_bb_[expert_id]->set_data(up_owner);
    down_bb_[expert_id]->set_data(down_owner);

    pack_bf16_buffer_b(gate_bb_[expert_id].get(), const_cast<ggml_bf16_t*>(gate_raw), config_.intermediate_size);
    pack_bf16_buffer_b(up_bb_[expert_id].get(), const_cast<ggml_bf16_t*>(up_raw), config_.intermediate_size);

    if (full_intermediate_size_ == config_.intermediate_size) {
      pack_bf16_buffer_b(down_bb_[expert_id].get(), const_cast<ggml_bf16_t*>(down_full_raw), config_.hidden_size);
    } else {
      std::vector<ggml_bf16_t> down_contiguous((size_t)config_.hidden_size * (size_t)config_.intermediate_size);
      const size_t down_col_offset = (size_t)tp_part_idx * (size_t)config_.intermediate_size;
      for (int row = 0; row < config_.hidden_size; ++row) {
        const ggml_bf16_t* src_row =
            down_full_raw + (size_t)row * (size_t)full_intermediate_size_ + down_col_offset;
        ggml_bf16_t* dst_row = down_contiguous.data() + (size_t)row * (size_t)config_.intermediate_size;
        std::memcpy(dst_row, src_row, sizeof(ggml_bf16_t) * (size_t)config_.intermediate_size);
      }
      pack_bf16_buffer_b(down_bb_[expert_id].get(), down_contiguous.data(), config_.hidden_size);
    }

    packed_gate_owner_[expert_id] = gate_owner;
    packed_up_owner_[expert_id] = up_owner;
    packed_down_owner_[expert_id] = down_owner;
    resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    expert_states_[expert_id].store(pin_after_pack ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, pin_after_pack);
    mesh::record_cache_promote(config_, tp_part_idx, expert_id);
    return true;
  }

  void decrement_active_load_count() {
    int current = active_load_count_.load(std::memory_order_acquire);
    while (current > 0 &&
           !active_load_count_.compare_exchange_weak(
               current, current - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
  }

  bool bf16_background_capacity_reached() const {
    if (cache_capacity_ <= 0) return true;
    return resident_expert_count_.load(std::memory_order_acquire) +
               active_load_count_.load(std::memory_order_acquire) >=
           cache_capacity_;
  }

  bool ensure_bf16_capacity_for_loading(int expert_id) {
    if (cache_capacity_ <= 0) return true;
    while (resident_expert_count_.load(std::memory_order_acquire) +
               active_load_count_.load(std::memory_order_acquire) >=
           cache_capacity_) {
      if (!evict_one_cached_expert(expert_id)) {
        break;
      }
    }
    return resident_expert_count_.load(std::memory_order_acquire) +
               active_load_count_.load(std::memory_order_acquire) <
           cache_capacity_;
  }

  ktransformers::AsyncExpertReader::ReadPriority read_priority_for_task(ExpertLoadSource source) const {
    return source == ExpertLoadSource::ColdStartWarmFill
               ? ktransformers::AsyncExpertReader::ReadPriority::Prefetch
               : ktransformers::AsyncExpertReader::ReadPriority::Demand;
  }

  void record_task_source_queued(ExpertLoadSource source) {
    auto* stats = mesh::enabled_cache_stats(config_);
    if (stats == nullptr) return;
    if (source == ExpertLoadSource::DeferredTopK) {
      stats->defer_cold_load_count.fetch_add(1, std::memory_order_relaxed);
    } else if (source == ExpertLoadSource::ColdStartWarmFill) {
      stats->coldstart_warmfill_queued_count.fetch_add(1, std::memory_order_relaxed);
    }
  }

  static constexpr int64_t background_schedule_key() {
    return mesh::ExpertLoadScheduler::background_schedule_key();
  }

  uint64_t enqueue_expert_load_task(int expert_id,
                                    int64_t schedule_key,
                                    ExpertLoadSource source,
                                    const std::vector<uint8_t>* protected_mask,
                                    uint64_t warmfill_epoch = 0) {
    const mesh::ExpertTaskEnqueueResult result =
        expert_load_scheduler_.enqueue(expert_id, schedule_key, source, protected_mask, warmfill_epoch);
    if (result.joined_warmfill_as_deferred) {
      if (auto* stats = mesh::enabled_cache_stats(config_)) {
        stats->defer_join_warmfill_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (result.created) {
      record_task_source_queued(source);
    }
    return result.task_id;
  }

  void fail_expert_load_task(uint64_t task_id, int expert_id, ExpertLoadState state) {
    expert_load_scheduler_.fail_task(task_id, expert_id, state);
    if (expert_id >= 0 && expert_id < static_cast<int>(pending_prefetches_.size()) &&
        pending_prefetches_[expert_id].task_id == task_id) {
      clear_pending_prefetch(expert_id, true);
    }
  }

  void promote_pending_task_for_demand(int expert_id, int64_t schedule_key = 0) {
    if (expert_id < 0 || expert_id >= static_cast<int>(pending_prefetches_.size())) return;
    const uint64_t task_id = pending_prefetches_[expert_id].task_id;
    if (task_id == 0) return;
    const mesh::ExpertTaskPromoteResult result =
        expert_load_scheduler_.promote_pending_for_demand(task_id, schedule_key);
    if (result.joined_warmfill) {
      if (auto* stats = mesh::enabled_cache_stats(config_)) {
        stats->demand_join_warmfill_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  bool promote_queued_task_for_deferred(int expert_id,
                                        int64_t schedule_key,
                                        const std::vector<uint8_t>* protected_mask) {
    const mesh::ExpertTaskPromoteResult result =
        expert_load_scheduler_.promote_queued_for_deferred(expert_id, schedule_key, protected_mask);
    if (result.joined_warmfill) {
      if (auto* stats = mesh::enabled_cache_stats(config_)) {
        stats->defer_join_warmfill_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (result.counted_deferred_cold) {
      if (auto* stats = mesh::enabled_cache_stats(config_)) {
        stats->defer_cold_load_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return result.promoted;
  }

  bool submit_expert_load_task(uint64_t task_id) {
    ExpertLoadTask task_snapshot;
    if (!expert_load_scheduler_.snapshot(task_id, &task_snapshot)) return false;
    if (task_snapshot.expert_id < 0 || task_snapshot.expert_id >= config_.expert_num) return false;
    if (!ensure_bf16_capacity_for_loading(task_snapshot.expert_id)) {
      fail_expert_load_task(task_id, task_snapshot.expert_id, ExpertLoadState::Canceled);
      expert_states_[task_snapshot.expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return false;
    }

    const Bf16IouringReadPlan plan = bf16_iouring_read_plan(task_snapshot.expert_id);
    void* gate_io = nullptr;
    void* up_io = nullptr;
    void* down_full_io = nullptr;
    if (!allocate_bf16_iouring_raw_buffers(plan, &gate_io, &up_io, &down_full_io)) {
      mesh::log_bf16_iouring_temp_alloc_failure(task_snapshot.expert_id, gate_io, up_io, down_full_io);
      fail_expert_load_task(task_id, task_snapshot.expert_id, ExpertLoadState::Failed);
      expert_states_[task_snapshot.expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return false;
    }

    try {
      std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
      read_batch.reserve(3);
      append_bf16_iouring_read_requests_for_expert(task_snapshot.expert_id,
                                                   plan,
                                                   gate_io,
                                                   up_io,
                                                   down_full_io,
                                                   read_priority_for_task(task_snapshot.source),
                                                   &read_batch);
      for (auto& request : read_batch) {
        request.schedule_key = task_snapshot.schedule_key;
      }
      std::vector<uint64_t> request_ids = config_.async_reader->submit_reads(read_batch);
      record_iouring_read_stats(read_batch);
      if (request_ids.empty()) {
        free_bf16_iouring_raw_buffers(plan, gate_io, up_io, down_full_io);
        fail_expert_load_task(task_id, task_snapshot.expert_id, ExpertLoadState::Failed);
        expert_states_[task_snapshot.expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        return false;
      }
      active_load_count_.fetch_add(1, std::memory_order_acq_rel);
      expert_load_scheduler_.mark_submitted(task_id, -1, gate_io, up_io, down_full_io, request_ids);
      auto& pending = pending_prefetches_[task_snapshot.expert_id];
      pending.slot_index = -1;
      pending.task_id = task_id;
      pending.requests = std::move(request_ids);
      if (task_snapshot.source == ExpertLoadSource::ColdStartWarmFill) {
        if (auto* stats = mesh::enabled_cache_stats(config_)) {
          stats->coldstart_warmfill_submitted_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
      mesh::record_cache_async_prefetch(config_);
      return true;
    } catch (...) {
      free_bf16_iouring_raw_buffers(plan, gate_io, up_io, down_full_io);
      fail_expert_load_task(task_id, task_snapshot.expert_id, ExpertLoadState::Failed);
      expert_states_[task_snapshot.expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      throw;
    }
  }

  void cancel_queued_warmfill_epoch(uint64_t epoch) {
    const mesh::ExpertTaskCancelResult result = expert_load_scheduler_.cancel_queued_warmfill_epoch(epoch);
    for (int expert_id : result.expert_ids) {
      if (expert_id >= 0 && expert_id < static_cast<int>(pending_prefetches_.size())) {
        pending_prefetches_[expert_id].task_id = 0;
        pending_prefetches_[expert_id].slot_index = -1;
        pending_prefetches_[expert_id].requests.clear();
      }
      if (expert_id >= 0 && expert_id < config_.expert_num) {
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      }
    }
    if (result.canceled > 0) {
      if (auto* stats = mesh::enabled_cache_stats(config_)) {
        stats->coldstart_warmfill_cancel_count.fetch_add(result.canceled, std::memory_order_relaxed);
      }
    }
  }

  void drain_expert_load_queue(bool allow_background) {
    constexpr int kMaxFiniteTasksPerDrain = 32;
    int submitted = 0;
    for (;;) {
      if (!allow_background && submitted >= kMaxFiniteTasksPerDrain) return;
      if (allow_background && bf16_background_capacity_reached()) {
        cancel_queued_warmfill_epoch(expert_load_scheduler_.current_coldstart_warmfill_epoch());
        return;
      }
      const uint64_t task_id = expert_load_scheduler_.pop_next(allow_background, bf16_background_capacity_reached());
      if (task_id == 0) return;
      ExpertLoadSource source = ExpertLoadSource::Explicit;
      uint64_t epoch = 0;
      ExpertLoadTask task_snapshot;
      if (expert_load_scheduler_.snapshot(task_id, &task_snapshot)) {
        source = task_snapshot.source;
        epoch = task_snapshot.warmfill_epoch;
      }
      if (!submit_expert_load_task(task_id) && source == ExpertLoadSource::ColdStartWarmFill) {
        cancel_queued_warmfill_epoch(epoch);
        return;
      }
      submitted += 1;
      if (source == ExpertLoadSource::ColdStartWarmFill && bf16_background_capacity_reached()) {
        cancel_queued_warmfill_epoch(epoch);
        return;
      }
    }
  }

  bool pending_prefetch_finished(int expert_id, bool* failed) const {
    return mesh::pending_prefetch_finished(pending_prefetches_[expert_id], config_.async_reader, failed);
  }

  void clear_pending_prefetch(int expert_id, bool release_buffers) {
    if (expert_id < 0 || expert_id >= static_cast<int>(pending_prefetches_.size())) return;
    auto& pending = pending_prefetches_[expert_id];
    const uint64_t task_id = pending.task_id;
    ExpertLoadTask task_snapshot;
    const bool have_task = task_id != 0 && expert_load_scheduler_.snapshot(task_id, &task_snapshot);
    if (release_buffers && have_task && task_snapshot.state == ExpertLoadState::Submitted) {
      try {
        free_bf16_iouring_raw_buffers_for_expert(
            expert_id, task_snapshot.gate_owner, task_snapshot.up_owner, task_snapshot.down_owner);
      } catch (...) {
      }
      decrement_active_load_count();
    }
    pending.slot_index = -1;
    pending.task_id = 0;
    pending.requests.clear();
    if (task_id != 0) {
      expert_load_scheduler_.mark_task_canceled_if_active(task_id);
      (void)expert_load_scheduler_.clear_active_if_matches(expert_id, task_id);
    }
  }

  bool finalize_pending_prefetch(int expert_id, bool pin_after_wait) {
    if (expert_id < 0 || expert_id >= static_cast<int>(pending_prefetches_.size())) return false;
    auto& pending = pending_prefetches_[expert_id];
    const uint64_t task_id = pending.task_id;
    ExpertLoadTask task_snapshot;
    if (task_id == 0 || !expert_load_scheduler_.snapshot(task_id, &task_snapshot) ||
        task_snapshot.gate_owner == nullptr || task_snapshot.up_owner == nullptr || task_snapshot.down_owner == nullptr) {
      clear_pending_prefetch(expert_id, true);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return false;
    }

    const Bf16IouringReadPlan plan = bf16_iouring_read_plan(expert_id);
    const auto* gate_raw =
        reinterpret_cast<const ggml_bf16_t*>(static_cast<const char*>(task_snapshot.gate_owner) + plan.gate_span.payload_shift);
    const auto* up_raw =
        reinterpret_cast<const ggml_bf16_t*>(static_cast<const char*>(task_snapshot.up_owner) + plan.up_span.payload_shift);
    const auto* down_full_raw = reinterpret_cast<const ggml_bf16_t*>(
        static_cast<const char*>(task_snapshot.down_owner) + plan.down_span.payload_shift);

    pending.slot_index = -1;
    pending.task_id = 0;
    pending.requests.clear();
    const bool ok = pack_iouring_expert_from_raw_slices(expert_id, gate_raw, up_raw, down_full_raw, pin_after_wait);
    free_bf16_iouring_raw_buffers(plan, task_snapshot.gate_owner, task_snapshot.up_owner, task_snapshot.down_owner);
    decrement_active_load_count();
    if (ok) {
      expert_load_scheduler_.mark_completed(task_id, expert_id);
      return true;
    }
    expert_load_scheduler_.fail_task(task_id, expert_id, ExpertLoadState::Failed);
    expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
    return false;
  }

  bool wait_for_pending_prefetch(int expert_id, bool pin_after_wait) {
    if (!iouring_enabled() || config_.async_reader == nullptr) return false;
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    auto& pending = pending_prefetches_[expert_id];
    if (pending.requests.empty() && pending.task_id != 0) {
      promote_pending_task_for_demand(expert_id);
      drain_expert_load_queue(false);
    }
    const auto& requests = pending_prefetches_[expert_id].requests;
    if (requests.empty()) return false;
    const int timeout_ms = 60000;
    if (!config_.async_reader->wait_for_requests(requests, timeout_ms)) {
      mesh::log_bf16_iouring_read_failure(expert_id, requests, config_.async_reader);
      clear_pending_prefetch(expert_id, true);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return false;
    }
    mesh::record_cache_prefetch_hit(config_, tp_part_idx, expert_id);
    return finalize_pending_prefetch(expert_id, pin_after_wait);
  }

  void complete_ready_prefetches() {
    if (!iouring_enabled() || config_.async_reader == nullptr) return;
    (void)config_.async_reader->poll_completions();
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      if (expert_states_[expert_id].load(std::memory_order_acquire) != EXPERT_PREFETCHING) continue;
      bool failed = false;
      if (pending_prefetch_finished(expert_id, &failed)) {
        finalize_pending_prefetch(expert_id, false);
      } else if (failed) {
        clear_pending_prefetch(expert_id, true);
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      }
    }
  }

  bool submit_async_prefetch(int expert_id,
                             const std::vector<uint8_t>* protected_mask = nullptr,
                             int64_t schedule_key = 0,
                             ExpertLoadSource source = ExpertLoadSource::DeferredTopK,
                             uint64_t warmfill_epoch = 0) {
    if (!iouring_enabled() || config_.async_reader == nullptr) return false;
    if (expert_id < 0 || expert_id >= config_.expert_num || config_.should_skip_expert(expert_id)) return false;
    uint8_t expected = EXPERT_BASELINE;
    if (!expert_states_[expert_id].compare_exchange_strong(
            expected, EXPERT_PREFETCHING, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return false;
    }
    const uint64_t task_id = enqueue_expert_load_task(expert_id, schedule_key, source, protected_mask, warmfill_epoch);
    if (task_id == 0) {
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return false;
    }
    auto& pending = pending_prefetches_[expert_id];
    pending.slot_index = -1;
    pending.task_id = task_id;
    pending.requests.clear();
    return true;
  }

  int prefetch_experts(int count,
                       const int64_t* expert_ids,
                       int protect_count = 0,
                       const int64_t* protect_ids = nullptr,
                       int max_to_submit = 0,
                       int prefetch_kind = 0,
                       int64_t schedule_key = 0) {
    if (!lazy_weight_enabled() || !iouring_enabled() || expert_ids == nullptr || count <= 0 || cache_capacity_ <= 0) {
      return 0;
    }
    complete_ready_prefetches();
    const bool is_background_warmfill = prefetch_kind == 1;
    const bool is_prefill_active = prefetch_kind == 2;
    const bool is_bootstrap_prefetch = is_background_warmfill;
    const ExpertLoadSource source = is_background_warmfill
                                        ? ExpertLoadSource::ColdStartWarmFill
                                        : (is_prefill_active ? ExpertLoadSource::PrefillActive
                                                             : ExpertLoadSource::DeferredTopK);
    uint64_t warmfill_epoch = 0;
    if (is_background_warmfill) {
      if (expert_load_scheduler_.coldstart_warmfill_submitted()) {
        if (auto* stats = mesh::enabled_cache_stats(config_)) {
          stats->speculative_prefetch_blocked_count.fetch_add(static_cast<uint64_t>(count), std::memory_order_relaxed);
        }
        return 0;
      }
      warmfill_epoch = expert_load_scheduler_.begin_coldstart_warmfill_epoch();
      schedule_key = background_schedule_key();
    }

    std::vector<uint8_t> protected_mask =
        mesh::build_protected_mask(config_, expert_ids, count, protect_ids, protect_count);
    std::vector<uint8_t> seen(config_.expert_num, 0);
    const int limit = max_to_submit > 0 ? max_to_submit : count;
    int submitted = 0;
    uint64_t bootstrap_candidate_count = 0;
    uint64_t bootstrap_skip_gpu_count = 0;
    uint64_t bootstrap_skip_resident_count = 0;
    for (int i = 0; i < count && submitted < limit; ++i) {
      const int expert_id = static_cast<int>(expert_ids[i]);
      if (expert_id < 0 || expert_id >= config_.expert_num) continue;
      if (config_.should_skip_expert(expert_id)) {
        if (is_bootstrap_prefetch && tp_part_idx == 0) bootstrap_skip_gpu_count += 1;
        continue;
      }
      if (seen[expert_id]) continue;
      seen[expert_id] = 1;
      if (is_bootstrap_prefetch && tp_part_idx == 0) bootstrap_candidate_count += 1;

      const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state != EXPERT_BASELINE) {
        if (!is_background_warmfill && state == EXPERT_PREFETCHING &&
            promote_queued_task_for_deferred(expert_id, schedule_key, &protected_mask)) {
          submitted += 1;
        }
        if (is_bootstrap_prefetch && tp_part_idx == 0) bootstrap_skip_resident_count += 1;
        continue;
      }
      if (submit_async_prefetch(expert_id, &protected_mask, schedule_key, source, warmfill_epoch)) {
        submitted += 1;
      }
    }
    if (is_background_warmfill) {
      mesh::record_cache_prefetch_submit(config_, static_cast<uint64_t>(submitted));
      drain_expert_load_queue(true);
    } else {
      drain_expert_load_queue(false);
    }
    mesh::record_bootstrap_prefetch_stats(config_,
                                          tp_part_idx,
                                          is_bootstrap_prefetch,
                                          bootstrap_candidate_count,
                                          static_cast<uint64_t>(submitted),
                                          bootstrap_skip_gpu_count,
                                          bootstrap_skip_resident_count);
    return submitted;
  }

  void prefetch_experts_binding(intptr_t expert_ids,
                                int count,
                                intptr_t protect_ids,
                                int protect_count,
                                int max_to_submit,
                                int prefetch_kind = 0,
                                int64_t schedule_key = 0) {
    prefetch_experts(count,
                     reinterpret_cast<const int64_t*>(expert_ids),
                     protect_count,
                     reinterpret_cast<const int64_t*>(protect_ids),
                     max_to_submit,
                     prefetch_kind,
                     schedule_key);
  }

  int prepare_prefill_layer_window(int64_t schedule_key = 0) {
    (void)schedule_key;
    complete_ready_prefetches();
    return 0;
  }

  bool release_prefill_layer_window() {
    complete_ready_prefetches();
    return true;
  }

  int transition_to_decode_cache(int requested_decode_capacity, int fill_limit) {
    (void)requested_decode_capacity;
    (void)fill_limit;
    complete_ready_prefetches();
    return 0;
  }

  bool allocate_and_pack_expert(int expert_id, bool pin_after_pack) {
    if (!iouring_enabled()) {
      return false;
    }
    const size_t resident_bytes_per_expert = gate_weight_bytes_ + up_weight_bytes_ + down_weight_bytes_;
    const size_t promotion_temp_bytes =
        sizeof(ggml_bf16_t) * (size_t)config_.hidden_size * (size_t)config_.intermediate_size;
    if (cache_capacity_ > 0) {
      while (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        if (!evict_one_cached_expert(expert_id)) {
          break;
        }
      }
      if (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        int cached_count = 0;
        int pinned_count = 0;
        int packing_count = 0;
        int demoting_count = 0;
        int busy_count = 0;
        for (int i = 0; i < config_.expert_num; ++i) {
          const uint8_t state = expert_states_[i].load(std::memory_order_acquire);
          if (state == EXPERT_CACHED) {
            ++cached_count;
            if (active_readers_[i].load(std::memory_order_acquire) != 0) ++busy_count;
          } else if (state == EXPERT_PINNED) {
            ++pinned_count;
          } else if (state == EXPERT_PACKING) {
            ++packing_count;
          } else if (state == EXPERT_DEMOTING) {
            ++demoting_count;
          }
        }
        mesh::log_bf16_cache_full_failure(config_,
                                          expert_id,
                                          resident_expert_count_.load(std::memory_order_acquire),
                                          cache_capacity_,
                                          cached_count,
                                          pinned_count,
                                          packing_count,
                                          demoting_count,
                                          busy_count,
                                          resident_bytes_per_expert,
                                          promotion_temp_bytes);
        return false;
      }
    }

    void* gate_owner = numa_alloc_onnode(gate_weight_bytes_, numa_node_);
    void* up_owner = numa_alloc_onnode(up_weight_bytes_, numa_node_);
    void* down_owner = numa_alloc_onnode(down_weight_bytes_, numa_node_);
    if (gate_owner == nullptr || up_owner == nullptr || down_owner == nullptr) {
      mesh::log_bf16_numa_alloc_failure(expert_id,
                                        gate_weight_bytes_,
                                        up_weight_bytes_,
                                        down_weight_bytes_,
                                        gate_owner,
                                        up_owner,
                                        down_owner,
                                        numa_node_);
      if (gate_owner) numa_free(gate_owner, gate_weight_bytes_);
      if (up_owner) numa_free(up_owner, up_weight_bytes_);
      if (down_owner) numa_free(down_owner, down_weight_bytes_);
      return false;
    }

    if (!read_and_pack_iouring_expert(expert_id, gate_owner, up_owner, down_owner)) {
      numa_free(gate_owner, gate_weight_bytes_);
      numa_free(up_owner, up_weight_bytes_);
      numa_free(down_owner, down_weight_bytes_);
      return false;
    }

    packed_gate_owner_[expert_id] = gate_owner;
    packed_up_owner_[expert_id] = up_owner;
    packed_down_owner_[expert_id] = down_owner;
    resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    expert_states_[expert_id].store(pin_after_pack ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, pin_after_pack);
    mesh::record_cache_promote(config_, tp_part_idx, expert_id);
    mesh::maybe_log_cpu_mem_resident(config_,
                                     tp_part_idx,
                                     expert_id,
                                     resident_expert_count_.load(std::memory_order_acquire),
                                     cache_capacity_,
                                     resident_bytes_per_expert,
                                     promotion_temp_bytes,
                                     pin_after_pack);
    return true;
  }

  void ensure_expert_ready(int expert_id, bool pin = false) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;

    mesh::record_cache_total_access(config_, tp_part_idx, expert_id);
    bool counted_miss_for_this_access = false;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PINNED) {
        if (!counted_miss_for_this_access) {
          mesh::record_cache_hit(config_, tp_part_idx, expert_id);
        }
        return;
      }
      if (state == EXPERT_CACHED) {
        if (!counted_miss_for_this_access) {
          mesh::record_cache_hit(config_, tp_part_idx, expert_id);
        }
        if (!pin) {
          return;
        }
        uint8_t expected = EXPERT_CACHED;
        if (expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PINNED, std::memory_order_acq_rel, std::memory_order_acquire)) {
          note_expert_pin(expert_id);
          return;
        }
        continue;
      }
      if (state == EXPERT_BASELINE) {
        mesh::record_cache_cold_miss(config_, tp_part_idx, expert_id);
        counted_miss_for_this_access = true;
        uint8_t expected = EXPERT_BASELINE;
        if (!expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PREFETCHING, std::memory_order_acq_rel, std::memory_order_acquire)) {
          continue;
        }
        const uint64_t task_id = enqueue_expert_load_task(expert_id, 0, ExpertLoadSource::Demand, nullptr, 0);
        if (task_id == 0) {
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          throw std::runtime_error("BF16 lazy-pack promotion failed");
        }
        auto& pending = pending_prefetches_[expert_id];
        pending.slot_index = -1;
        pending.task_id = task_id;
        pending.requests.clear();
        drain_expert_load_queue(false);
        if (!wait_for_pending_prefetch(expert_id, pin)) {
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          throw std::runtime_error("BF16 lazy-pack promotion failed");
        }
        continue;
      }
      if (state == EXPERT_PREFETCHING) {
        mesh::record_cache_inflight_miss(config_, tp_part_idx, expert_id);
        counted_miss_for_this_access = true;
        if (!wait_for_pending_prefetch(expert_id, pin)) {
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          throw std::runtime_error("BF16 lazy-pack promotion failed");
        }
        continue;
      }
      if (state == EXPERT_PACKING || state == EXPERT_DEMOTING) {
        std::this_thread::yield();
        continue;
      }
    }
  }
#endif
