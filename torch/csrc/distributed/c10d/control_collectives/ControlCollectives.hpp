#pragma once

#include <ATen/core/ivalue.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <c10/macros/Macros.h>
#include <torch/custom_class.h>

namespace c10d {

using namespace std::chrono_literals;

class TORCH_API ControlCollectives : public torch::CustomClassHolder {
 public:
  virtual void barrier(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min,
      bool block = true) = 0;

  virtual void broadcast_send(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  virtual std::vector<uint8_t> broadcast_recv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual void gather_send(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  virtual std::vector<std::vector<uint8_t>> gather_recv(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual std::vector<uint8_t> scatter_send(
      const std::string& key,
      const std::vector<std::vector<uint8_t>>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  virtual std::vector<uint8_t> scatter_recv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual std::vector<std::vector<uint8_t>> all_gather(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual int64_t all_sum(
      const std::string& key,
      int64_t data,
      std::chrono::milliseconds timeout = 5min) = 0;
};

} // namespace c10d
