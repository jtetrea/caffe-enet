#ifndef CAFFE_DENSE_MM_IMAGE_DATA_LAYER_HPP_
#define CAFFE_DENSE_MM_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_transformer.hpp"


namespace caffe {


template <typename Dtype>
class DenseMMImageDataLayer : public BasePrefetchingMMDataLayer<Dtype> {
 public:
  explicit DenseMMImageDataLayer(const LayerParameter& param)
      : BasePrefetchingMMDataLayer<Dtype>(param) {}
  virtual ~DenseMMImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseMMImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  vector <vector <std::string> > lines_;
  int lines_id_;
  Blob<Dtype> transformed_label_;
  

};


}  // namespace caffe

#endif  
