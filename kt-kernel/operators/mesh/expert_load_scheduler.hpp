#ifndef CPUINFER_OPERATOR_MESH_EXPERT_LOAD_SCHEDULER_HPP
#define CPUINFER_OPERATOR_MESH_EXPERT_LOAD_SCHEDULER_HPP

#include <cstdint>
#include <algorithm>
#include <limits>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "expert_residency.hpp"

#ifndef _WIN32

namespace mesh {

struct ExpertTaskEnqueueResult {
  uint64_t task_id = 0;
  bool created = false;
  bool updated = false;
  bool joined_warmfill_as_deferred = false;
  ExpertLoadSource queued_source = ExpertLoadSource::Explicit;
};

struct ExpertTaskPromoteResult {
  bool promoted = false;
  bool joined_warmfill = false;
  bool counted_deferred_cold = false;
};

struct ExpertTaskCancelResult {
  std::vector<int> expert_ids;
  uint64_t canceled = 0;
};

class ExpertLoadScheduler {
 public:
  static constexpr int64_t background_schedule_key() {
    return std::numeric_limits<int64_t>::max();
  }

  void reset(int expert_num) {
    std::lock_guard<std::mutex> guard(mu_);
    tasks_.clear();
    active_task_by_expert_.assign(std::max(0, expert_num), 0);
    queue_ = Queue{};
    queued_warmfill_tasks_.clear();
    next_task_id_ = 1;
    next_sequence_ = 1;
    coldstart_warmfill_epoch_ = 0;
    coldstart_warmfill_submitted_ = false;
  }

  void clear_keep_capacity(int expert_num) {
    std::lock_guard<std::mutex> guard(mu_);
    tasks_.clear();
    active_task_by_expert_.assign(std::max(0, expert_num), 0);
    queue_ = Queue{};
    queued_warmfill_tasks_.clear();
    coldstart_warmfill_submitted_ = false;
  }

  bool coldstart_warmfill_submitted() const {
    std::lock_guard<std::mutex> guard(mu_);
    return coldstart_warmfill_submitted_;
  }

  uint64_t begin_coldstart_warmfill_epoch() {
    std::lock_guard<std::mutex> guard(mu_);
    coldstart_warmfill_submitted_ = true;
    return ++coldstart_warmfill_epoch_;
  }

  uint64_t current_coldstart_warmfill_epoch() const {
    std::lock_guard<std::mutex> guard(mu_);
    return coldstart_warmfill_epoch_;
  }

  ExpertTaskEnqueueResult enqueue(int expert_id,
                                  int64_t schedule_key,
                                  ExpertLoadSource source,
                                  const std::vector<uint8_t>* protected_mask,
                                  uint64_t warmfill_epoch = 0) {
    std::lock_guard<std::mutex> guard(mu_);
    ExpertTaskEnqueueResult result;
    result.queued_source = source;
    if (expert_id < 0 || expert_id >= static_cast<int>(active_task_by_expert_.size())) return result;

    const uint64_t existing_id = active_task_by_expert_[expert_id];
    if (existing_id != 0) {
      auto it = tasks_.find(existing_id);
      if (it != tasks_.end()) {
        ExpertLoadTask& task = it->second;
        if (task.state == ExpertLoadState::Queued) {
          result.task_id = existing_id;
          if (schedule_key < task.schedule_key || task.source == ExpertLoadSource::ColdStartWarmFill) {
            result.joined_warmfill_as_deferred =
                task.source == ExpertLoadSource::ColdStartWarmFill && source == ExpertLoadSource::DeferredTopK;
            task.schedule_key = schedule_key;
            task.source = source;
            task.warmfill_epoch = warmfill_epoch;
            task.queue_generation += 1;
            if (protected_mask != nullptr) {
              task.protected_mask = *protected_mask;
            }
            queue_.push(ExpertLoadQueueEntry{
                task.schedule_key, task.sequence, task.queue_generation, task.task_id});
            result.updated = true;
          }
          return result;
        }
        if (task.state == ExpertLoadState::Submitting || task.state == ExpertLoadState::Submitted) {
          result.task_id = existing_id;
          return result;
        }
      }
      active_task_by_expert_[expert_id] = 0;
    }

    const uint64_t task_id = next_task_id_++;
    ExpertLoadTask task;
    task.task_id = task_id;
    task.expert_id = expert_id;
    task.schedule_key = schedule_key;
    task.source = source;
    task.state = ExpertLoadState::Queued;
    task.queue_generation = 1;
    task.sequence = next_sequence_++;
    task.warmfill_epoch = warmfill_epoch;
    if (protected_mask != nullptr) {
      task.protected_mask = *protected_mask;
    }
    tasks_[task_id] = std::move(task);
    active_task_by_expert_[expert_id] = task_id;
    queue_.push(ExpertLoadQueueEntry{schedule_key, tasks_[task_id].sequence, 1, task_id});
    if (source == ExpertLoadSource::ColdStartWarmFill) {
      queued_warmfill_tasks_.push_back(task_id);
    }
    result.task_id = task_id;
    result.created = true;
    return result;
  }

  ExpertTaskCancelResult cancel_queued_warmfill_epoch(uint64_t epoch) {
    std::lock_guard<std::mutex> guard(mu_);
    ExpertTaskCancelResult result;
    for (uint64_t task_id : queued_warmfill_tasks_) {
      auto it = tasks_.find(task_id);
      if (it == tasks_.end()) continue;
      ExpertLoadTask& task = it->second;
      if (task.source != ExpertLoadSource::ColdStartWarmFill ||
          task.warmfill_epoch != epoch ||
          task.state != ExpertLoadState::Queued) {
        continue;
      }
      task.state = ExpertLoadState::Canceled;
      task.queue_generation += 1;
      if (task.expert_id >= 0 && task.expert_id < static_cast<int>(active_task_by_expert_.size()) &&
          active_task_by_expert_[task.expert_id] == task_id) {
        active_task_by_expert_[task.expert_id] = 0;
      }
      result.expert_ids.push_back(task.expert_id);
      result.canceled += 1;
    }
    return result;
  }

  ExpertTaskPromoteResult promote_pending_for_demand(uint64_t task_id) {
    std::lock_guard<std::mutex> guard(mu_);
    ExpertTaskPromoteResult result;
    if (task_id == 0) return result;
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return result;
    ExpertLoadTask& task = it->second;
    if (task.state != ExpertLoadState::Queued) return result;
    result.joined_warmfill = task.source == ExpertLoadSource::ColdStartWarmFill;
    task.schedule_key = 0;
    task.source = ExpertLoadSource::Demand;
    task.queue_generation += 1;
    queue_.push(ExpertLoadQueueEntry{task.schedule_key, task.sequence, task.queue_generation, task.task_id});
    result.promoted = true;
    return result;
  }

  ExpertTaskPromoteResult promote_queued_for_deferred(int expert_id,
                                                       int64_t schedule_key,
                                                       const std::vector<uint8_t>* protected_mask) {
    std::lock_guard<std::mutex> guard(mu_);
    ExpertTaskPromoteResult result;
    if (expert_id < 0 || expert_id >= static_cast<int>(active_task_by_expert_.size())) return result;
    const uint64_t task_id = active_task_by_expert_[expert_id];
    if (task_id == 0) return result;
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return result;
    ExpertLoadTask& task = it->second;
    if (task.state != ExpertLoadState::Queued) return result;
    if (schedule_key >= task.schedule_key && task.source != ExpertLoadSource::ColdStartWarmFill) {
      result.promoted = true;
      return result;
    }
    const bool was_warmfill = task.source == ExpertLoadSource::ColdStartWarmFill;
    task.schedule_key = schedule_key;
    task.source = ExpertLoadSource::DeferredTopK;
    task.warmfill_epoch = 0;
    task.queue_generation += 1;
    if (protected_mask != nullptr) {
      task.protected_mask = *protected_mask;
    }
    queue_.push(ExpertLoadQueueEntry{task.schedule_key, task.sequence, task.queue_generation, task.task_id});
    result.promoted = true;
    result.joined_warmfill = was_warmfill;
    result.counted_deferred_cold = was_warmfill;
    return result;
  }

  void fail_task(uint64_t task_id, int expert_id, ExpertLoadState state) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
      it->second.state = state;
      it->second.queue_generation += 1;
    }
    clear_active_locked(expert_id, task_id);
  }

  uint64_t pop_next(bool allow_background, bool background_capacity_reached) {
    std::lock_guard<std::mutex> guard(mu_);
    while (!queue_.empty()) {
      const ExpertLoadQueueEntry entry = queue_.top();
      auto it = tasks_.find(entry.task_id);
      if (it == tasks_.end()) {
        queue_.pop();
        continue;
      }
      ExpertLoadTask& task = it->second;
      if (task.state != ExpertLoadState::Queued ||
          task.queue_generation != entry.queue_generation ||
          task.schedule_key != entry.schedule_key) {
        queue_.pop();
        continue;
      }
      if (task.schedule_key == background_schedule_key()) {
        if (!allow_background || background_capacity_reached) return 0;
      }
      queue_.pop();
      task.state = ExpertLoadState::Submitting;
      return task.task_id;
    }
    return 0;
  }

  bool snapshot(uint64_t task_id, ExpertLoadTask* out) const {
    if (out == nullptr) return false;
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return false;
    *out = it->second;
    return true;
  }

  void mark_submitted(uint64_t task_id, int slot, void* gate_owner, void* up_owner, void* down_owner,
                      const std::vector<uint64_t>& requests) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return;
    ExpertLoadTask& task = it->second;
    task.slot = slot;
    task.gate_owner = gate_owner;
    task.up_owner = up_owner;
    task.down_owner = down_owner;
    task.request_ids = requests;
    task.state = ExpertLoadState::Submitted;
  }

  void mark_task_canceled_if_active(uint64_t task_id) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tasks_.find(task_id);
    if (it != tasks_.end() &&
        (it->second.state == ExpertLoadState::Queued ||
         it->second.state == ExpertLoadState::Submitting ||
         it->second.state == ExpertLoadState::Submitted)) {
      it->second.state = ExpertLoadState::Canceled;
      it->second.queue_generation += 1;
    }
  }

  void mark_completed(uint64_t task_id, int expert_id) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
      it->second.state = ExpertLoadState::Completed;
    }
    clear_active_locked(expert_id, task_id);
  }

  ExpertLoadSource task_source_or(uint64_t task_id, ExpertLoadSource fallback) const {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tasks_.find(task_id);
    return it == tasks_.end() ? fallback : it->second.source;
  }

  uint64_t task_epoch_or(uint64_t task_id, uint64_t fallback) const {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tasks_.find(task_id);
    return it == tasks_.end() ? fallback : it->second.warmfill_epoch;
  }

  bool clear_active_if_matches(int expert_id, uint64_t task_id) {
    std::lock_guard<std::mutex> guard(mu_);
    return clear_active_locked(expert_id, task_id);
  }

 private:
  using Queue = std::priority_queue<ExpertLoadQueueEntry,
                                    std::vector<ExpertLoadQueueEntry>,
                                    ExpertLoadQueueEntryCompare>;

  bool clear_active_locked(int expert_id, uint64_t task_id) {
    if (expert_id >= 0 && expert_id < static_cast<int>(active_task_by_expert_.size()) &&
        active_task_by_expert_[expert_id] == task_id) {
      active_task_by_expert_[expert_id] = 0;
      return true;
    }
    return false;
  }

  mutable std::mutex mu_;
  std::unordered_map<uint64_t, ExpertLoadTask> tasks_;
  std::vector<uint64_t> active_task_by_expert_;
  Queue queue_;
  std::vector<uint64_t> queued_warmfill_tasks_;
  uint64_t next_task_id_ = 1;
  uint64_t next_sequence_ = 1;
  uint64_t coldstart_warmfill_epoch_ = 0;
  bool coldstart_warmfill_submitted_ = false;
};

}  // namespace mesh

#endif  // _WIN32

#endif  // CPUINFER_OPERATOR_MESH_EXPERT_LOAD_SCHEDULER_HPP
