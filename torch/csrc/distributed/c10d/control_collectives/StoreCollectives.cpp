#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/StoreCollectives.hpp>
#include <chrono>
#include <exception>
#include <vector>

namespace {
std::string getRankKey(const std::string& key, int rank) {
  return fmt::format("{}/{}", key, rank);
}
} // namespace

namespace c10d {

StoreCollectives::StoreCollectives(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int world_size)
    : store_(std::move(store)), rank_(rank), world_size_(world_size) {}

void StoreCollectives::barrier(
    const std::string& key,
    std::chrono::milliseconds timeout,
    bool blocking) {
  StoreTimeoutGuard g{*store_, timeout};

  auto num_members_key = fmt::format("{}/num_members", key);
  auto last_members_key = fmt::format("{}/last_members", key);

  auto idx = store_->add(num_members_key, 1);
  store_->set(getRankKey(key, rank_), "joined");

  if (idx == world_size_) {
    store_->set(last_members_key, "<val_ignored>");
  } else if (blocking) {
    try {
      store_->wait({last_members_key});
    } catch (const std::exception& e) {
      std::string msg = "barrier failed -- missing ranks: ";
      for (int i = 0; i < world_size_; i++) {
        if (i == rank_) {
          continue;
        }
        auto rank_key = getRankKey(key, i);
        if (!store_->check({rank_key})) {
          msg += fmt::format("{}, ", i);
        }
      }
      throw std::runtime_error(msg + e.what());
    }
  }
}

void StoreCollectives::broadcast_send(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  store_->set(key, data);
}

std::vector<uint8_t> StoreCollectives::broadcast_recv(
    const std::string& key,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  return store_->get(key);
}

void StoreCollectives::gather_send(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  auto rank_key = getRankKey(key, rank_);
  store_->set(rank_key, data);
}

std::vector<std::vector<uint8_t>> StoreCollectives::gather_recv(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  std::vector<std::string> keys;
  keys.reserve(world_size_);

  for (int i = 0; i < world_size_; i++) {
    if (i == rank_) {
      continue;
    }
    auto rank_key = getRankKey(key, i);
    keys.emplace_back(rank_key);
  }

  std::vector<std::vector<uint8_t>> results;
  results.reserve(world_size_);

  try {
    results = store_->multiGet(keys);
  } catch (const std::exception& e) {
    std::string msg = "gather failed -- missing ranks: ";
    for (int i = 0; i < world_size_; i++) {
      if (i == rank_) {
        continue;
      }
      auto rank_key = getRankKey(key, i);
      if (!store_->check({rank_key})) {
        msg += fmt::format("{}, ", i);
      }
    }
    throw std::runtime_error(msg + e.what());
  }

  // insert local data
  results.insert(results.begin() + rank_, data);
  return results;
}

std::vector<uint8_t> StoreCollectives::scatter_send(
    const std::string& key,
    const std::vector<std::vector<uint8_t>>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  std::vector<std::string> keys;
  keys.reserve(world_size_);
  for (int i = 0; i < world_size_; i++) {
    if (i == rank_) {
      continue;
    }
    auto rank_key = getRankKey(key, i);
    keys.emplace_back(rank_key);
  }
  auto local = data.at(rank_);

  std::vector<std::vector<uint8_t>> to_send{data};

  to_send.erase(to_send.begin() + rank_);

  store_->multiSet(keys, to_send);

  return local;
}

std::vector<uint8_t> StoreCollectives::scatter_recv(
    const std::string& key,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  auto rank_key = getRankKey(key, rank_);
  return store_->get(rank_key);
}

std::vector<std::vector<uint8_t>> StoreCollectives::all_gather(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  auto local_key = getRankKey(key, rank_);
  store_->set(local_key, data);

  std::vector<std::string> keys;
  keys.reserve(world_size_);

  for (int i = 0; i < world_size_; i++) {
    auto rank_key = getRankKey(key, i);
    keys.emplace_back(rank_key);
  }

  try {
    return store_->multiGet(keys);
  } catch (const std::exception& e) {
    std::string msg = "all_gather failed -- missing ranks: ";
    for (int i = 0; i < world_size_; i++) {
      if (i == rank_) {
        continue;
      }
      auto rank_key = getRankKey(key, i);
      if (!store_->check({rank_key})) {
        msg += fmt::format("{}, ", i);
      }
    }
    throw std::runtime_error(msg + e.what());
  }
}

int64_t StoreCollectives::all_sum(
    const std::string& key,
    int64_t value,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  store_->add(key, value);

  barrier(key, timeout);

  return store_->add(key, 0);
}

} // namespace c10d
