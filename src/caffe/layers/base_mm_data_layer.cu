#include <vector>
#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingMMDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  // Reshape to loaded depth
  top[1]->ReshapeLike(this->prefetch_depth_);
  // Copy the data
  caffe_copy(prefetch_depth_.count(), prefetch_depth_.cpu_data(),
      top[1]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[2]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingMMDataLayer);

}  // namespace caffe
