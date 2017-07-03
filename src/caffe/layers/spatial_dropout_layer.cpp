// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/spatial_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.spatial_dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int channels = bottom[0]->shape(1); // number of channels
  const int height = bottom[0]->shape(2);
  const int width = bottom[0]->shape(3);
  if (this->phase_ == TRAIN || this->layer_param_.spatial_dropout_param().sample_weights_test()) {
    // Create random numbers
    caffe_rng_bernoulli(channels, 1. - threshold_, mask);
    for (int i = 0; i < channels; ++i) {
      for (int j = 0; j <  height * width; ++j) {
        top_data[i * j] = bottom_data[i * j] * mask[i] * scale_;
      }
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN || this->layer_param_.spatial_dropout_param().sample_weights_test()) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int channels = bottom[0]->shape(1); // number of channels
      const int height = bottom[0]->shape(2);
      const int width = bottom[0]->shape(3);
      for (int i = 0; i < channels; ++i) {
        for (int j = 0; j <  height * width; ++j) {
          bottom_diff[i * j] = top_diff[i * j] * mask[i] * scale_;
        }
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SpatialDropoutLayer);
#endif

INSTANTIATE_CLASS(SpatialDropoutLayer);
REGISTER_LAYER_CLASS(SpatialDropout);

}  // namespace caffe
