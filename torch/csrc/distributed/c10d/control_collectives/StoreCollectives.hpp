#pragma once

#include <c10/macros/Macros.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/ControlCollectives.hpp>

namespace c10d {

class TORCH_API StoreCollectives : public ControlCollectives {
 public:
  explicit StoreCollectives(
      c10::intrusive_ptr<Store> store,
      int rank,
      int world_size);

  void barrier(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min,
      bool block = true) override;

  void broadcast_send(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<uint8_t> broadcast_recv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) override;

  void gather_send(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<std::vector<uint8_t>> gather_recv(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;

  std::vector<uint8_t> scatter_send(
      const std::string& key,
      const std::vector<std::vector<uint8_t>>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<uint8_t> scatter_recv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) override;

  std::vector<std::vector<uint8_t>> all_gather(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;

  int64_t all_sum(
      const std::string& key,
      int64_t data,
      std::chrono::milliseconds timeout = 5min) override;

 private:
  c10::intrusive_ptr<Store> store_;
  int rank_;
  int world_size_;
};

} // namespace c10d
