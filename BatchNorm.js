const tf2 = require('@tensorflow/tfjs');
var _ = require('underscore');

class Normalizer {
  constructor(offset, name) {
    this.offset = offset;
    this.name = name;
  }
}

export default class BatchNorm extends Normalizer{
    constructor(offset, name) {
        super( offset, name);
    }
  // 
  norm(inputs) {
    // return tf2.batchNorm(inputs, tf2.tensor(), tf2.tensor(), this.offset);
    const batchNormLayer = tf2.layers.batchNormalization({
      epsilon: 1e-5,
      scale: false,
      center: this.offset,
      name: this.name
  })

  return batchNormLayer.apply(inputs);

    // return tf2.conv2d(inputs, this.kernal_shape, this.channel_size, 1, this.name);
  }
}