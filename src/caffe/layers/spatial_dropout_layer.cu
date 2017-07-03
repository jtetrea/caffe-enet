#include <vector>

#include "caffe/layers/spatial_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SpatialDropoutForward(const int n, const int h, const int w, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    int current_channel = index % (h * w);
    out[index] = in[index] * (mask[current_channel] > threshold) * scale;
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int channels = bottom[0]->shape(1); // number of channels
  const int height = bottom[0]->shape(2);
  const int width = bottom[0]->shape(3);
  if (this->phase_ == TRAIN || this->layer_param_.spatial_dropout_param().sample_weights_test()) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());

    caffe_gpu_rng_uniform(channels, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpatialDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, height, width, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void SpatialDropoutBackward(const int n, const int h, const int w, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    int current_channel = index % (h * w);
    out_diff[index] = in_diff[index] * (mask[current_channel] > threshold) * scale;
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN || this->layer_param_.spatial_dropout_param().sample_weights_test()) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      const int channels = bottom[0]->shape(1); // number of channels
      const int height = bottom[0]->shape(2);
      const int width = bottom[0]->shape(3);
      // NOLINT_NEXT_LINE(whitespace/operators)
      SpatialDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, height, width, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialDropoutLayer);

}  // namespace caffe
