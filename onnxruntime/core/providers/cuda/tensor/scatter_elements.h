// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class ScatterElements final : public CudaKernel {
 public:
  ScatterElements(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");
  }
  ~ScatterElements() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime
