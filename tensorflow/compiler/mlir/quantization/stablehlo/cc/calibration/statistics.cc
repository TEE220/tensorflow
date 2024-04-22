/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/statistics.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/min_max_value.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"

namespace stablehlo::quantization {
namespace {

using ::stablehlo::quantization::CalibrationOptions;
using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::calibrator::CalibrationStatisticsMap;
using ::tensorflow::quantization::PyFunctionLibrary;

// Reads the calibration statistics from the given directory.
absl::Status ReadStatistics(
    absl::string_view calibration_data_dir,
    absl::flat_hash_map<std::string, CalibrationStatistics>& statistics_map) {
  std::vector<std::string> statistics_files;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(
      std::string(calibration_data_dir), &statistics_files));

  for (const std::string& statistics_file : statistics_files) {
    CalibrationStatisticsMap single_map;
    TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(
        tsl::Env::Default(),
        tsl::io::JoinPath(calibration_data_dir, statistics_file), &single_map));
    statistics_map.insert(single_map.statistics().begin(),
                          single_map.statistics().end());
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status AddCalibrationStatistics(
    mlir::ModuleOp module_op, absl::string_view calibration_data_dir,
    const CalibrationOptions& calibration_options,
    const PyFunctionLibrary& py_function_library) {
  absl::flat_hash_map<std::string, CalibrationStatistics> statistics_map;
  TF_RETURN_IF_ERROR(ReadStatistics(calibration_data_dir, statistics_map));

  absl::Status status = absl::OkStatus();
  module_op.walk([&py_function_library, &calibration_options, &status,
                  &statistics_map](mlir::TF::CustomAggregatorOp aggregator_op) {
    mlir::StringRef id = aggregator_op.getId();
    auto iter = statistics_map.find(id);
    if (iter == statistics_map.end()) {
      status = absl::InternalError(
          absl::StrFormat("Calibrated data does not exist. Cannot find "
                          "statistics. value for id: %s",
                          id));
      return;
    }

    const std::optional<MinMaxValue> min_max_values =
        py_function_library.GetCalibrationMinMaxValue(iter->second,
                                                      calibration_options);
    if (min_max_values == std::nullopt) {
      status = absl::InternalError(
          "Cannot find min/max values for calibration statistics.");
      return;
    }

    const auto [min_value, max_value] = *min_max_values;
    mlir::OpBuilder builder(aggregator_op);
    aggregator_op->setAttr("min", builder.getF32FloatAttr(min_value));
    aggregator_op->setAttr("max", builder.getF32FloatAttr(max_value));
  });
  return status;
}

}  // namespace stablehlo::quantization
