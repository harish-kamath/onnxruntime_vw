//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

//scikit-learn is a Python module for machine learning built on top of SciPy and
//distributed under the 3-Clause BSD license. See https://github.com/scikit-learn/scikit-learn.
//This material is licensed under the BSD License (see https://github.com/scikit-learn/scikit-learn/blob/master/COPYING);
/* Modifications Copyright (c) Microsoft. */

#include "contrib_ops/cpu/vw/GDPredict.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    GDPredict,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<uint32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>(),
                                                      DataTypeImpl::GetTensorType<uint64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()
}),
    GDPredict);

Status GDPredict::Compute(OpKernelContext* ctx) const {
    const Tensor* features = ctx->Input<Tensor>(0);
    
    const TensorShape& input_shape = features->Shape();
    std::vector<int64_t> output_dims = {1};
    Tensor* output_tensor = ctx->Output(0, TensorShape(output_dims));
    
    
    auto X_Data = (features->Data<uint32_t>());
    auto Y_Data = (output_tensor->MutableData<float>());
    
    
    
    for(int64_t i = 0; i < input_shape.Size(); i++){
        //std::cout << "Multiplying feature: " << features[i] << " by weight: " << weights[i] << "\n";
        Y_Data[0] +=  weights[X_Data[i]];
        //std::cout << "Current summation: " << out[0] << "\n\n";
    }
    
    return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
