// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class GDPredict final : public OpKernel {
 public:
  GDPredict(const OpKernelInfo& info) : OpKernel(info) {
      
      std::string weights_string = info.GetAttrOrDefault<std::string>("weights",std::string("_Unused"));
      // If possible, always prefer std::vector to naked array

       // Build an istream that holds the input string
       std::istringstream iss(weights_string);

       // Iterate over the istream, using >> to grab floats
       // and push_back to store them in the vector
       std::copy(std::istream_iterator<float>(iss),
             std::istream_iterator<float>(),
             std::back_inserter(weights));

       // Put the result on standard out
      /*
      std::copy(weights.begin(), weights.end(),
             std::ostream_iterator<float>(std::cout, ","));
       std::cout << "\n";
       */
  }

  Status Compute(OpKernelContext* context) const override;

private :
    std::vector<float> weights;
};
}  // namespace contrib
}  // namespace onnxruntime
