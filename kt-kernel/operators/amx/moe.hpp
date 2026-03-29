/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_H
#define CPUINFER_OPERATOR_AMX_MOE_H

// #define CHECK
// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#ifndef _WIN32
#include <numa.h>
#include <sys/mman.h>
#endif

#include <atomic>
#include <thread>

#include "moe_base.hpp"

template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 private:
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

#ifndef _WIN32
  enum ExpertState : uint8_t {
    EXPERT_BASELINE = 0,
    EXPERT_PACKING = 1,
    EXPERT_CACHED = 2,
    EXPERT_PINNED = 3,
    EXPERT_DEMOTING = 4,
  };

  std::vector<void*> baseline_gate_weight_src_;
  std::vector<void*> baseline_up_weight_src_;
  std::vector<void*> baseline_down_weight_src_;
  std::vector<float*> baseline_gate_scale_src_;
  std::vector<float*> baseline_up_scale_src_;
  std::vector<float*> baseline_down_scale_src_;
  std::vector<float*> baseline_gate_mins_src_;
  std::vector<float*> baseline_up_mins_src_;
  std::vector<float*> baseline_down_mins_src_;

  std::vector<void*> packed_gate_owner_;
  std::vector<void*> packed_up_owner_;
  std::vector<void*> packed_down_owner_;

  std::unique_ptr<std::atomic<uint8_t>[]> expert_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> active_readers_;

  std::atomic<int> resident_expert_count_{0};
  std::atomic<int> eviction_cursor_{0};
  int numa_node_ = 0;
  int cache_capacity_ = 0;
  size_t gate_total_bytes_ = 0;
  size_t up_total_bytes_ = 0;
  size_t down_total_bytes_ = 0;
  size_t gate_weight_bytes_ = 0;
  size_t up_weight_bytes_ = 0;
  size_t down_weight_bytes_ = 0;
  size_t gate_scale_bytes_ = 0;
  size_t up_scale_bytes_ = 0;
  size_t down_scale_bytes_ = 0;
  size_t gate_mins_bytes_ = 0;
  size_t up_mins_bytes_ = 0;
  size_t down_mins_bytes_ = 0;
#endif

#ifdef CHECK
  char verify_bb[100000000];
  char check_bb[100000000];
  uint8_t compare_expers = 3;
#endif

  inline void write_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                            size_t scale_size) {
    // printf("expert %d, size %ld, scale size %ld\n", expert_idx, size, scale_size);
    // std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                               std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file: %s", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                            std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                     .c_str());
      // throw std::runtime_error("No such file");
    }
    of.write((char*)bb, size - scale_size);
    of.close();
    // of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                      "_scale_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file\n");
      // throw std::runtime_error("No such file");
    }
    of.write(((char*)bb) + size - scale_size, scale_size);
  }

  inline void read_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                           size_t scale_size, uint8_t mat_split, uint8_t mat_split_idex) {
    // std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * (size - scale_size) / mat_split);
    f.read(((char*)bb) + mat_split_idex * (size - scale_size) / mat_split, (size - scale_size) / mat_split);
    f.close();
    // f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                     "_scale_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(scale_size) + "Byte" + "_scale_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * scale_size / mat_split);
    f.read((((char*)bb) + size - scale_size) + mat_split_idex * scale_size / mat_split, scale_size / mat_split);
  }
#ifdef CHECK
  inline void load_check() {
    memcpy(check_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
  }

  void verify_load_right() {
    // printf("varify down bb_0 %d\n", tp_part_idx);
    memcpy(verify_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
    // check if verify_bb_0 equal to check_bb_0
    if (memcmp(verify_bb, check_bb, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size)) != 0) {
      printf("verify error\n");
      for (size_t i = 0; i < T::BufferB::required_size(config_.hidden_size, config_.intermediate_size); ++i) {
        if (verify_bb[i] != check_bb[i]) {
          printf("Difference at byte %zu: verify_bb_%d[%zu] = %02x, check_bb[%zu] = %02x\n", i, compare_expers, i,
                 (unsigned char)verify_bb[i], i, (unsigned char)check_bb[i]);
          break;  // find the first difference and exit
        }
      }
      assert(0);
    } else {
      printf("pass verify\n");
      // pick out the 100th~150th byte of scale to see
      printf("numa %d, verify_bb_%d:\n", tp_part_idx, compare_expers);
      size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
      size_t scale_size = config_.hidden_size * sizeof(float);
      for (size_t i = size - scale_size; i < size - scale_size + 50; ++i) {
        printf("%02x ", (unsigned char)verify_bb[i]);
      }
      printf("\n");
    }
  }
#endif

#ifdef FORWARD_TIME_REPORT
  std::chrono::time_point<std::chrono::high_resolution_clock> last_now;
#endif

 public:
  AMX_MOE_TP() = default;

  AMX_MOE_TP(GeneralMOEConfig config, int tp_part_idx = 0) : Base(config, tp_part_idx) {
    // Initialization now happens in derived_init() which is called by base constructor
  }

  void derived_init() {
    printf("Creating AMX_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
#ifndef _WIN32
    if (config_.use_mmap) {
      initialize_lazy_mmap_state();
    }
#endif
    auto& load = config_.load;
    auto& save = config_.save;

    std::filesystem::path prefix = config_.path;
    prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
    if (save) {
      std::cout << "Creating " << prefix << std::endl;
      std::filesystem::create_directories(prefix);
    }
    if (load) {
      if (std::filesystem::exists(prefix)) {
        std::cout << "Loading from " << prefix << std::endl;
      } else {
        throw std::runtime_error("Path not found: " + prefix.string());
      }
    }
  }

  ~AMX_MOE_TP() {
#ifndef _WIN32
    if (config_.use_mmap) {
      for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
        free_packed_expert(expert_id);
      }
    }
#endif
  }

#ifndef _WIN32
  void initialize_lazy_mmap_state() {
    const int en = config_.expert_num;
    baseline_gate_weight_src_.assign(en, nullptr);
    baseline_up_weight_src_.assign(en, nullptr);
    baseline_down_weight_src_.assign(en, nullptr);
    baseline_gate_scale_src_.assign(en, nullptr);
    baseline_up_scale_src_.assign(en, nullptr);
    baseline_down_scale_src_.assign(en, nullptr);
    baseline_gate_mins_src_.assign(en, nullptr);
    baseline_up_mins_src_.assign(en, nullptr);
    baseline_down_mins_src_.assign(en, nullptr);
    packed_gate_owner_.assign(en, nullptr);
    packed_up_owner_.assign(en, nullptr);
    packed_down_owner_.assign(en, nullptr);
    expert_states_ = std::make_unique<std::atomic<uint8_t>[]>(en);
    active_readers_ = std::make_unique<std::atomic<uint32_t>[]>(en);
    resident_expert_count_.store(0, std::memory_order_relaxed);
    eviction_cursor_.store(0, std::memory_order_relaxed);

    gate_total_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    up_total_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    down_total_bytes_ = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
    gate_weight_bytes_ = reinterpret_cast<const char*>(gate_bb_[0]->d) - reinterpret_cast<const char*>(gate_bb_[0]->b);
    up_weight_bytes_ = reinterpret_cast<const char*>(up_bb_[0]->d) - reinterpret_cast<const char*>(up_bb_[0]->b);
    down_weight_bytes_ = reinterpret_cast<const char*>(down_bb_[0]->d) - reinterpret_cast<const char*>(down_bb_[0]->b);
    if constexpr (requires { gate_bb_[0]->mins; }) {
      gate_scale_bytes_ = reinterpret_cast<const char*>(gate_bb_[0]->mins) - reinterpret_cast<const char*>(gate_bb_[0]->d);
      up_scale_bytes_ = reinterpret_cast<const char*>(up_bb_[0]->mins) - reinterpret_cast<const char*>(up_bb_[0]->d);
      down_scale_bytes_ = reinterpret_cast<const char*>(down_bb_[0]->mins) - reinterpret_cast<const char*>(down_bb_[0]->d);
      gate_mins_bytes_ = gate_total_bytes_ - gate_weight_bytes_ - gate_scale_bytes_;
      up_mins_bytes_ = up_total_bytes_ - up_weight_bytes_ - up_scale_bytes_;
      down_mins_bytes_ = down_total_bytes_ - down_weight_bytes_ - down_scale_bytes_;
    } else {
      gate_scale_bytes_ = gate_total_bytes_ - gate_weight_bytes_;
      up_scale_bytes_ = up_total_bytes_ - up_weight_bytes_;
      down_scale_bytes_ = down_total_bytes_ - down_weight_bytes_;
    }
    cache_capacity_ = config_.max_tier0_experts <= 0
                          ? 0
                          : std::min(config_.expert_num, std::max(config_.max_tier0_experts, config_.num_experts_per_tok));

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
      gate_bb_[expert_id]->d = nullptr;
      up_bb_[expert_id]->d = nullptr;
      down_bb_[expert_id]->d = nullptr;
      if constexpr (requires { gate_bb_[expert_id]->mins; }) {
        gate_bb_[expert_id]->mins = nullptr;
        up_bb_[expert_id]->mins = nullptr;
        down_bb_[expert_id]->mins = nullptr;
      }
    }
  }

  void apply_baseline_ptrs(int expert_id) {
    gate_bb_[expert_id]->b = reinterpret_cast<decltype(gate_bb_[expert_id]->b)>(baseline_gate_weight_src_[expert_id]);
    up_bb_[expert_id]->b = reinterpret_cast<decltype(up_bb_[expert_id]->b)>(baseline_up_weight_src_[expert_id]);
    down_bb_[expert_id]->b = reinterpret_cast<decltype(down_bb_[expert_id]->b)>(baseline_down_weight_src_[expert_id]);
    gate_bb_[expert_id]->d = baseline_gate_scale_src_[expert_id];
    up_bb_[expert_id]->d = baseline_up_scale_src_[expert_id];
    down_bb_[expert_id]->d = baseline_down_scale_src_[expert_id];
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      gate_bb_[expert_id]->mins = baseline_gate_mins_src_[expert_id];
      up_bb_[expert_id]->mins = baseline_up_mins_src_[expert_id];
      down_bb_[expert_id]->mins = baseline_down_mins_src_[expert_id];
    }
  }

  void apply_owned_ptrs(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
    gate_bb_[expert_id]->b = reinterpret_cast<decltype(gate_bb_[expert_id]->b)>(gate_owner);
    up_bb_[expert_id]->b = reinterpret_cast<decltype(up_bb_[expert_id]->b)>(up_owner);
    down_bb_[expert_id]->b = reinterpret_cast<decltype(down_bb_[expert_id]->b)>(down_owner);
    gate_bb_[expert_id]->d = reinterpret_cast<float*>(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_);
    up_bb_[expert_id]->d = reinterpret_cast<float*>(reinterpret_cast<char*>(up_owner) + up_weight_bytes_);
    down_bb_[expert_id]->d = reinterpret_cast<float*>(reinterpret_cast<char*>(down_owner) + down_weight_bytes_);
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      gate_bb_[expert_id]->mins = reinterpret_cast<float*>(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_ + gate_scale_bytes_);
      up_bb_[expert_id]->mins = reinterpret_cast<float*>(reinterpret_cast<char*>(up_owner) + up_weight_bytes_ + up_scale_bytes_);
      down_bb_[expert_id]->mins = reinterpret_cast<float*>(reinterpret_cast<char*>(down_owner) + down_weight_bytes_ + down_scale_bytes_);
    }
  }

  void set_mmap_source_ptrs_quantized(int expert_id, void* gate_weight_src, void* up_weight_src, void* down_weight_src,
                                      float* gate_scale_src, float* up_scale_src, float* down_scale_src,
                                      float* gate_mins_src = nullptr, float* up_mins_src = nullptr,
                                      float* down_mins_src = nullptr) {
    baseline_gate_weight_src_[expert_id] = gate_weight_src;
    baseline_up_weight_src_[expert_id] = up_weight_src;
    baseline_down_weight_src_[expert_id] = down_weight_src;
    baseline_gate_scale_src_[expert_id] = gate_scale_src;
    baseline_up_scale_src_[expert_id] = up_scale_src;
    baseline_down_scale_src_[expert_id] = down_scale_src;
    baseline_gate_mins_src_[expert_id] = gate_mins_src;
    baseline_up_mins_src_[expert_id] = up_mins_src;
    baseline_down_mins_src_[expert_id] = down_mins_src;
    apply_baseline_ptrs(expert_id);
    expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
  }

  void acquire_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_add(1, std::memory_order_acq_rel);
  }

  void release_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_sub(1, std::memory_order_acq_rel);
  }

  class ExpertReadScope {
   public:
    ExpertReadScope(AMX_MOE_TP* owner, std::vector<int> expert_ids) : owner_(owner), experts_(std::move(expert_ids)) {
      for (int expert_id : experts_) {
        owner_->acquire_expert_read(expert_id);
      }
    }

    ~ExpertReadScope() {
      for (int expert_id : experts_) {
        owner_->release_expert_read(expert_id);
      }
    }

   private:
    AMX_MOE_TP* owner_;
    std::vector<int> experts_;
  };

  void free_packed_expert(int expert_id) {
    void* gate_owner = packed_gate_owner_[expert_id];
    void* up_owner = packed_up_owner_[expert_id];
    void* down_owner = packed_down_owner_[expert_id];
    packed_gate_owner_[expert_id] = nullptr;
    packed_up_owner_[expert_id] = nullptr;
    packed_down_owner_[expert_id] = nullptr;

    if (gate_owner) {
      numa_free(gate_owner, gate_total_bytes_);
    }
    if (up_owner) {
      numa_free(up_owner, up_total_bytes_);
    }
    if (down_owner) {
      numa_free(down_owner, down_total_bytes_);
    }
  }

  bool evict_one_cached_expert(int exclude_expert_id) {
    if (config_.expert_num <= 1) return false;

    for (int attempt = 0; attempt < config_.expert_num; ++attempt) {
      const int victim = (eviction_cursor_.fetch_add(1, std::memory_order_acq_rel) + attempt) % config_.expert_num;
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
      apply_baseline_ptrs(victim);
      resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      expert_states_[victim].store(EXPERT_BASELINE, std::memory_order_release);
      return true;
    }

    return false;
  }

  bool allocate_and_copy_expert(int expert_id, bool pin_after_copy) {
    if (cache_capacity_ > 0) {
      while (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        if (!evict_one_cached_expert(expert_id)) {
          break;
        }
      }
    }

    void* gate_owner = numa_alloc_onnode(gate_total_bytes_, numa_node_);
    void* up_owner = numa_alloc_onnode(up_total_bytes_, numa_node_);
    void* down_owner = numa_alloc_onnode(down_total_bytes_, numa_node_);
    if (gate_owner == nullptr || up_owner == nullptr || down_owner == nullptr) {
      if (gate_owner) numa_free(gate_owner, gate_total_bytes_);
      if (up_owner) numa_free(up_owner, up_total_bytes_);
      if (down_owner) numa_free(down_owner, down_total_bytes_);
      return false;
    }

    if (baseline_gate_weight_src_[expert_id] == nullptr || baseline_up_weight_src_[expert_id] == nullptr ||
        baseline_down_weight_src_[expert_id] == nullptr || baseline_gate_scale_src_[expert_id] == nullptr ||
        baseline_up_scale_src_[expert_id] == nullptr || baseline_down_scale_src_[expert_id] == nullptr) {
      numa_free(gate_owner, gate_total_bytes_);
      numa_free(up_owner, up_total_bytes_);
      numa_free(down_owner, down_total_bytes_);
      return false;
    }

    madvise(baseline_gate_weight_src_[expert_id], gate_weight_bytes_, MADV_WILLNEED);
    madvise(baseline_up_weight_src_[expert_id], up_weight_bytes_, MADV_WILLNEED);
    madvise(baseline_down_weight_src_[expert_id], down_weight_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_gate_scale_src_[expert_id], gate_scale_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_up_scale_src_[expert_id], up_scale_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_down_scale_src_[expert_id], down_scale_bytes_, MADV_WILLNEED);
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      if (baseline_gate_mins_src_[expert_id] != nullptr) {
        madvise((void*)baseline_gate_mins_src_[expert_id], gate_mins_bytes_, MADV_WILLNEED);
      }
      if (baseline_up_mins_src_[expert_id] != nullptr) {
        madvise((void*)baseline_up_mins_src_[expert_id], up_mins_bytes_, MADV_WILLNEED);
      }
      if (baseline_down_mins_src_[expert_id] != nullptr) {
        madvise((void*)baseline_down_mins_src_[expert_id], down_mins_bytes_, MADV_WILLNEED);
      }
    }

    std::memcpy(gate_owner, baseline_gate_weight_src_[expert_id], gate_weight_bytes_);
    std::memcpy(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_, baseline_gate_scale_src_[expert_id], gate_scale_bytes_);
    std::memcpy(up_owner, baseline_up_weight_src_[expert_id], up_weight_bytes_);
    std::memcpy(reinterpret_cast<char*>(up_owner) + up_weight_bytes_, baseline_up_scale_src_[expert_id], up_scale_bytes_);
    std::memcpy(down_owner, baseline_down_weight_src_[expert_id], down_weight_bytes_);
    std::memcpy(reinterpret_cast<char*>(down_owner) + down_weight_bytes_, baseline_down_scale_src_[expert_id], down_scale_bytes_);

    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      if (gate_mins_bytes_ > 0 && baseline_gate_mins_src_[expert_id] != nullptr) {
        std::memcpy(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_ + gate_scale_bytes_,
                    baseline_gate_mins_src_[expert_id], gate_mins_bytes_);
      }
      if (up_mins_bytes_ > 0 && baseline_up_mins_src_[expert_id] != nullptr) {
        std::memcpy(reinterpret_cast<char*>(up_owner) + up_weight_bytes_ + up_scale_bytes_,
                    baseline_up_mins_src_[expert_id], up_mins_bytes_);
      }
      if (down_mins_bytes_ > 0 && baseline_down_mins_src_[expert_id] != nullptr) {
        std::memcpy(reinterpret_cast<char*>(down_owner) + down_weight_bytes_ + down_scale_bytes_,
                    baseline_down_mins_src_[expert_id], down_mins_bytes_);
      }
    }

    packed_gate_owner_[expert_id] = gate_owner;
    packed_up_owner_[expert_id] = up_owner;
    packed_down_owner_[expert_id] = down_owner;
    apply_owned_ptrs(expert_id, gate_owner, up_owner, down_owner);
    resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    expert_states_[expert_id].store(pin_after_copy ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    return true;
  }

  void ensure_expert_ready(int expert_id, bool pin = false) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PINNED) {
        return;
      }
      if (state == EXPERT_CACHED) {
        if (!pin) {
          return;
        }
        uint8_t expected = EXPERT_CACHED;
        if (expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PINNED, std::memory_order_acq_rel, std::memory_order_acquire)) {
          return;
        }
        continue;
      }
      if (state == EXPERT_BASELINE) {
        uint8_t expected = EXPERT_BASELINE;
        if (!expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PACKING, std::memory_order_acq_rel, std::memory_order_acquire)) {
          continue;
        }
        if (!allocate_and_copy_expert(expert_id, pin)) {
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          throw std::runtime_error("AMX lazy-copy promotion failed");
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

  // ============================================================================
  // CRTP buffer creation - no group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = down_ba_[expert_idx];
    auto& bb = down_bb_[expert_idx];
    auto& bc = down_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    }
  }
  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    if (config_.gate_projs.size()) {
      if (config_.use_mmap) {
        // mmap baseline mode: experts start on file-backed pointers and can be
        // promoted into NUMA-local resident buffers on demand.
        pool->do_work_stealing_job(
            config_.expert_num, nullptr,
            [this, physical_to_logical_map](int expert_id) {
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
              float* gate_mins = nullptr;
              float* up_mins = nullptr;
              float* down_mins = nullptr;
              if constexpr (requires { gate_bb_[expert_id]->mins; }) {
                if (!config_.gate_zeros.empty()) gate_mins = (float*)(config_.gate_zeros[tp_part_idx][logical_expert_id]);
                if (!config_.up_zeros.empty()) up_mins = (float*)(config_.up_zeros[tp_part_idx][logical_expert_id]);
                if (!config_.down_zeros.empty()) down_mins = (float*)(config_.down_zeros[tp_part_idx][logical_expert_id]);
              }
              set_mmap_source_ptrs_quantized(
                  expert_id, config_.gate_projs[tp_part_idx][logical_expert_id],
                  config_.up_projs[tp_part_idx][logical_expert_id], config_.down_projs[tp_part_idx][logical_expert_id],
                  (float*)config_.gate_scales[tp_part_idx][logical_expert_id],
                  (float*)config_.up_scales[tp_part_idx][logical_expert_id],
                  (float*)config_.down_scales[tp_part_idx][logical_expert_id], gate_mins, up_mins, down_mins);
            },
            nullptr);
      } else {
        // Legacy copy mode
        pool->do_work_stealing_job(
            config_.expert_num, nullptr,
            [this, physical_to_logical_map](int expert_id) {
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
              {
                size_t scale_size = config_.intermediate_size * sizeof(float);
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) - scale_size;

                memcpy(gate_bb_[expert_id]->b, config_.gate_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(gate_bb_[expert_id]->d, config_.gate_scales[tp_part_idx][logical_expert_id], scale_size);
                }

                memcpy(up_bb_[expert_id]->b, config_.up_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(up_bb_[expert_id]->d, config_.up_scales[tp_part_idx][logical_expert_id], scale_size);
                }
              }

              {
                size_t scale_size = config_.hidden_size * sizeof(float);
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size) - scale_size;

                memcpy(down_bb_[expert_id]->b, config_.down_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(down_bb_[expert_id]->d, config_.down_scales[tp_part_idx][logical_expert_id], scale_size);
                }
              }
            },
            nullptr);
      }

    } else {
      int nth = T::recommended_nth(config_.intermediate_size);
      static uint8_t mat_type_all = 3, mat_split = 1;
      std::filesystem::path prefix = config_.path;
      prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));

      if (config_.load) {
        std::cout << "Loading from " << prefix << std::endl;
        for (int task_id = 0; task_id < config_.expert_num * mat_type_all * mat_split; task_id++) {
          int64_t expert_idx = task_id / (mat_type_all * mat_split);
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          uint8_t mat_class = (task_id % (mat_type_all * mat_split)) / mat_split;
          uint8_t mat_split_idex = task_id % mat_split;
          if (mat_class == 0) {  // the up matrix
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
            size_t scale_size = config_.intermediate_size * sizeof(float);
            read_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, logical_expert_id, size, scale_size, mat_split,
                         mat_split_idex);
          } else if (mat_class == 1) {
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
            size_t scale_size = config_.intermediate_size * sizeof(float);
            read_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                         mat_split, mat_split_idex);
          } else {
            size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
            size_t scale_size = config_.hidden_size * sizeof(float);
            read_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                         mat_split, mat_split_idex);
          }
        }
      }
// check process, store down matrix to check
#ifdef CHECK
      load_check();
#endif
#ifndef CHECK
      else
#endif
      {
        if (tp_part_idx == 0) {
          std::cout << "  online quant from bf16" << std::endl;
        }
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              // gate part
              gate_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.gate_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth);
              // up part
              up_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.up_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth);
            },
            nullptr);

        nth = T::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              // down part
              down_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.down_proj + logical_expert_id * config_.hidden_size * config_.intermediate_size,
                  ith, nth);
              // printf("load idown, expert %ld, ith %d, total nth %d\n", expert_idx, ith, nth);
            },
            nullptr);
      }
#ifdef CHECK
      verify_load_right();
#endif
      // save process
      if (config_.save) {
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all, nullptr,
            [this, physical_to_logical_map, prefix](int task_id) {
              int64_t expert_idx = task_id / mat_type_all;
              expert_idx = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = task_id % mat_type_all;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 2) {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                write_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, expert_idx, size, scale_size);
              }
            },
            nullptr);
      }
    }
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
#ifndef _WIN32
    std::unique_ptr<ExpertReadScope> expert_read_scope;
    std::vector<int> active_experts;
    if (config_.use_mmap) {
      active_experts.reserve(qlen * k);
      std::vector<uint8_t> seen(config_.expert_num, 0);

      for (int i = 0; i < qlen * k; ++i) {
        const int expert_id = (int)expert_ids[i];
        if (config_.should_skip_expert(expert_id) || seen[expert_id]) {
          continue;
        }
        seen[expert_id] = 1;
        active_experts.push_back(expert_id);
      }

      expert_read_scope = std::make_unique<ExpertReadScope>(this, active_experts);
      for (int expert_id : active_experts) {
        ensure_expert_ready(expert_id, false);
      }
    }
#endif
    Base::forward(qlen, k, expert_ids, weights, input, output);
#ifndef _WIN32
    if (config_.use_mmap && config_.max_tier0_experts <= 0 && !active_experts.empty()) {
      expert_read_scope.reset();
      for (int expert_id : active_experts) {
        demote_expert(expert_id);
      }
    }
#endif
  }

  void promote_expert(int expert_id) {
#ifndef _WIN32
    ensure_expert_ready(expert_id, true);
#else
    (void)expert_id;
#endif
  }

  void demote_expert(int expert_id) {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_BASELINE) {
        return;
      }
      if (state == EXPERT_PACKING) {
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
      apply_baseline_ptrs(expert_id);
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
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    return expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_PINNED;
#else
    (void)expert_id;
    return false;
#endif
  }

  // forward, forward_prefill, forward_decode, warm_up are inherited from Base
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_TP
// Inherits from TP_MOE<AMX_MOE_BASE<...>> to reuse merge_results implementation
// ============================================================================

template <typename K>
class TP_MOE<AMX_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else if (config.gate_proj != nullptr) {
      printf("From BF16\n");
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        if (tps[i]->config_.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
                memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                             col * tpc.intermediate_size,
                         (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                             col * config.intermediate_size + i * tpc.intermediate_size,
                         sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr);
        }
      }

      DO_TPS_LOAD_WEIGHTS(pool);

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }

      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file %s\n", config.path.c_str());
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else {
      throw std::runtime_error("no weight source");
    }
  }

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

  // merge_results is inherited from TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>>
};

#endif
