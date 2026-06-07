// This file is included inside TP_MOE<AMX_BF16_MOE_TP<K>>.
// Keep declarations free of namespace wrappers and include guards.

  void promote_expert(int expert_id) {
    if (this->tp_count <= 1) {
      this->tps[0]->promote_expert(expert_id);
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this, expert_id](int tp_id) { this->tps[tp_id]->promote_expert(expert_id); });
  }

  void demote_expert(int expert_id) {
    for (int i = 0; i < this->tp_count; ++i) {
      this->tps[i]->demote_expert(expert_id);
    }
  }

  bool is_expert_promoted(int expert_id) const {
    return !this->tps.empty() && this->tps[0]->is_expert_promoted(expert_id);
  }

  int expert_state_code(int expert_id) const {
    return !this->tps.empty() ? this->tps[0]->expert_state_code(expert_id) : 0;
  }

  void split_deferred_experts_binding(intptr_t source_ids,
                                      intptr_t immediate_ids,
                                      intptr_t deferred_ids,
                                      int count,
                                      int k,
                                      int max_deferred_per_token,
                                      int64_t defer_schedule_key = 0) {
    split_deferred_experts(reinterpret_cast<const int64_t*>(source_ids),
                           reinterpret_cast<int64_t*>(immediate_ids),
                           reinterpret_cast<int64_t*>(deferred_ids),
                           count,
                           k,
                           max_deferred_per_token,
                           defer_schedule_key);
  }

  void split_deferred_experts(const int64_t* source_ids,
                              int64_t* immediate_ids,
                              int64_t* deferred_ids,
                              int count,
                              int k,
                              int max_deferred_per_token,
                              int64_t defer_schedule_key = 0) {
    if (source_ids == nullptr || immediate_ids == nullptr || deferred_ids == nullptr || count <= 0 || k <= 0) return;

    const int defer_limit = std::max(0, std::min(max_deferred_per_token, k));
    std::vector<int64_t> ids(count);
    for (int i = 0; i < count; ++i) {
      ids[i] = source_ids[i];
      immediate_ids[i] = -1;
      deferred_ids[i] = -1;
    }

    if (!this->weights_loaded || this->tps.empty()) {
      for (int i = 0; i < count; ++i) {
        if (!this->config.should_skip_expert(ids[i])) {
          immediate_ids[i] = ids[i];
        }
      }
      return;
    }

    if (this->tp_count <= 1) {
      this->tps[0]->complete_ready_prefetches();
    } else {
      this->config.pool->dispense_backend()->do_numa_job(
          [this](int tp_id) { this->tps[tp_id]->complete_ready_prefetches(); });
    }

    std::vector<int64_t> cold_deferred;
    std::vector<int64_t> protected_immediate;
    cold_deferred.reserve(count);
    protected_immediate.reserve(count);

    const int rows = (count + k - 1) / k;
    mesh::StateDeferStats defer_stats;
    defer_stats.token_count = static_cast<uint64_t>(rows);
    for (int row = 0; row < rows; ++row) {
      int deferred_in_row = 0;
      int overflow_in_row = 0;
      const int row_begin = row * k;
      const int row_end = std::min(count, row_begin + k);
      for (int idx = row_begin; idx < row_end; ++idx) {
        const int expert_id = static_cast<int>(ids[idx]);
        if (this->config.should_skip_expert(expert_id)) {
          defer_stats.gpu_skip_count += 1;
          continue;
        }
        defer_stats.cpu_topk_count += 1;

        const bool ready = this->tps[0]->is_expert_ready_resident(expert_id);
        if (!ready) {
          defer_stats.nonready_count += 1;
        }
        if (!ready && deferred_in_row < defer_limit) {
          deferred_ids[idx] = expert_id;
          deferred_in_row += 1;
          defer_stats.deferred_count += 1;
          if (this->tps[0]->is_expert_cold_baseline(expert_id)) {
            cold_deferred.push_back(expert_id);
          }
          continue;
        }

        immediate_ids[idx] = expert_id;
        protected_immediate.push_back(expert_id);
        if (!ready) {
          defer_stats.overflow_immediate_count += 1;
          overflow_in_row += 1;
        }
      }
      if (overflow_in_row > 0) {
        defer_stats.overflow_token_count += 1;
      }
    }

    mesh::record_state_defer_stats(this->config, defer_stats);

    if (!cold_deferred.empty()) {
      prefetch_experts(static_cast<int>(cold_deferred.size()),
                       cold_deferred.data(),
                       static_cast<int>(protected_immediate.size()),
                       protected_immediate.empty() ? nullptr : protected_immediate.data(),
                       static_cast<int>(cold_deferred.size()),
                       0,
                       defer_schedule_key);
    }
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

  void prefetch_experts(int count,
                        const int64_t* expert_ids,
                        int protect_count = 0,
                        const int64_t* protect_ids = nullptr,
                        int max_to_submit = 0,
                        int prefetch_kind = 0,
                        int64_t schedule_key = 0) {
    if (!this->weights_loaded || expert_ids == nullptr || count <= 0) return;
    if (this->tp_count <= 1) {
      this->tps[0]->prefetch_experts(count, expert_ids, protect_count, protect_ids, max_to_submit, prefetch_kind,
                                     schedule_key);
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this, count, expert_ids, protect_count, protect_ids, max_to_submit, prefetch_kind, schedule_key](int tp_id) {
          this->tps[tp_id]->prefetch_experts(count, expert_ids, protect_count, protect_ids, max_to_submit,
                                             prefetch_kind, schedule_key);
        });
  }

  void mesh_prepare_prefill_layer_binding(int64_t schedule_key = 0) {
    if (!this->weights_loaded) return;
    if (this->tp_count <= 1) {
      this->tps[0]->prepare_prefill_layer_window(schedule_key);
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this, schedule_key](int tp_id) { this->tps[tp_id]->prepare_prefill_layer_window(schedule_key); });
  }

  void mesh_release_prefill_layer_binding() {
    if (!this->weights_loaded) return;
    if (this->tp_count <= 1) {
      (void)this->tps[0]->release_prefill_layer_window();
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this](int tp_id) { (void)this->tps[tp_id]->release_prefill_layer_window(); });
  }

  void mesh_transition_decode_cache_binding(int decode_capacity, int fill_limit) {
    if (!this->weights_loaded) return;
    if (this->tp_count <= 1) {
      this->tps[0]->transition_to_decode_cache(decode_capacity, fill_limit);
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this, decode_capacity, fill_limit](int tp_id) {
          this->tps[tp_id]->transition_to_decode_cache(decode_capacity, fill_limit);
        });
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0, x1;
          avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e), &x0, &x1);
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0);
          *((__m512*)(merge_to + e + 16)) = _mm512_add_ps(*((__m512*)(merge_to + e + 16)), x1);
        }
      }
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), *((__m512*)(merge_from + e)));
        }
      }
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
      }
    };

    auto pool = config.pool;
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, merge_fn);
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
