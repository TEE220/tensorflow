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

#include "xla/python/pgle_session.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/aggregate_profile.h"
#include "xla/python/xplane_to_profile_instructions.h"
#include "tsl/profiler/lib/profiler_session.h"

namespace xla {
std::unique_ptr<tsl::ProfilerSession> PGLESession::Trace() {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_python_tracer_level(1);
  options.set_enable_hlo_proto(true);
  return tsl::ProfilerSession::Create(options);
}

void PGLESession::StopTrace(
    std::unique_ptr<tsl::ProfilerSession> profiler_session) {
  tensorflow::profiler::XSpace xspace;
  // Disables the ProfilerSession
  xla::ThrowIfError(profiler_session->CollectData(&xspace));
  tensorflow::profiler::ProfiledInstructionsProto fdo_profile;
  xla::ThrowIfError(xla::ConvertXplaneToProfiledInstructionsProto(
      {std::move(xspace)}, &fdo_profile));
  fdo_profiles_.push_back(std::move(fdo_profile));
}

std::string PGLESession::GetFdoProfile() const {
  tensorflow::profiler::ProfiledInstructionsProto result_proto;
  xla::AggregateProfiledInstructionsProto(fdo_profiles_, &result_proto);
  return result_proto.SerializeAsString();
}

void PGLESessionRunner::Start() {
  if (pgle_data_collecting_retries_ <= 0 ||
      call_times_ > pgle_data_collecting_retries_) {
    return;
  }

  if (current_session_ != nullptr) {
    LOG(WARNING) << "PGLE profile session already started";
    return;
  }

  ++call_times_;
  current_session_ = pgle_session_.Trace();
}

void PGLESessionRunner::Stop() {
  if (current_session_ == nullptr) {
    LOG(WARNING) << "PGLE profile session not started";
    return;
  }

  pgle_session_.StopTrace(std::move(current_session_));
  current_session_ = nullptr;
}

bool PGLESessionRunner::IsFdoConsumed() {
  return pgle_data_collecting_retries_ == 0 || collected_fdo_.has_value();
}

std::optional<std::string> PGLESessionRunner::ConsumeFdoProfile() {
  if (collected_fdo_.has_value()) {
    return collected_fdo_;
  }

  if (pgle_data_collecting_retries_ <= 0 ||
      call_times_ != pgle_data_collecting_retries_) {
    return std::nullopt;
  }

  collected_fdo_ = pgle_session_.GetFdoProfile();
  return collected_fdo_;
}
}  // namespace xla
