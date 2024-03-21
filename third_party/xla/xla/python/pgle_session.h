/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PYTHON_PGLE_SESSION_H_
#define XLA_PYTHON_PGLE_SESSION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {
// Allows to run profile sessions several times and collect FDO results after.
class PGLESession {
 public:
  std::unique_ptr<tsl::ProfilerSession> Trace();

  void StopTrace(std::unique_ptr<tsl::ProfilerSession> profiler_session);

  std::string GetFdoProfile() const;

 private:
  std::vector<tensorflow::profiler::ProfiledInstructionsProto> fdo_profiles_;
};

class PGLESessionRunner {
 public:
  explicit PGLESessionRunner(int64_t pgle_data_collecting_retries)
      : pgle_data_collecting_retries_(pgle_data_collecting_retries) {}

  // Start profile session.
  void Start();

  // Stop profile session and aggregate the result.
  void Stop();

  // FDO profile will be returned only in case when profiles were collected
  // specified amount of times. Method result is cached.
  std::optional<std::string> ConsumeFdoProfile();

  // Check is fdo profile were consumed by client.
  bool IsFdoConsumed();

 private:
  int64_t pgle_data_collecting_retries_;
  int64_t call_times_ = 0;
  PGLESession pgle_session_;
  std::optional<std::string> collected_fdo_;
  std::unique_ptr<tsl::ProfilerSession> current_session_ = nullptr;
};
}  // namespace xla

#endif  // XLA_PYTHON_PGLE_SESSION_H_
