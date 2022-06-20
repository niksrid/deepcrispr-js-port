import * as tf2 from '@tensorflow/tfjs';
import * as _ from 'underscore';

class Encoder {
  constructor(channel_size, kernal_shape, name, strides, padding) {
    this.channel_size = channel_size;
    this.kernal_shape = kernal_shape;
    this.name = name;
    this.strides = strides;
    this.padding = padding || 'same';
  }
}

export class Conv2D extends Encoder{
    constructor(channel_size, kernal_shape, name, strides, padding) {
        super(channel_size, kernal_shape, name, strides, padding);
    }

    conv(inputs, strides = this.strides) {
      const convObj = {
        kernelSize: this.kernal_shape,
        filters: this.channel_size,
        activation: 'relu',
        padding: this.padding,
        kernelInitializer: 'VarianceScaling',
        name: this.name
      }
      if(strides) {
      convObj.strides = strides;
      }
      // console.log(convObj.filters, inputs.shape.reduce((a,c)=>a+c,0))
      // if(convObj.filters > inputs.shape[3]) {
      //   convObj.filters = inputs.shape[3];
      // }
      const conv2dLayer = tf2.layers.conv2d(convObj);
     const tensor =  conv2dLayer.apply(inputs);
     return tensor;
    }
}


class Normalizer {
  constructor(offset, name, decay) {
    this.offset = offset;
    this.name = name;
    this.decay = decay;
  }
}

export class BatchNorm extends Normalizer{
    constructor(offset, name, decay) {
        super( offset, name, decay);
    }
  // 
  norm(inputs) {
    // return tf2.batchNorm(inputs, tf2.tensor(), tf2.tensor(), this.offset);
    const batchNormLayer = tf2.layers.batchNormalization({
      epsilon: 1e-5,
      scale: false,
      center: this.offset,
      name: this.name,
      momentum: this.decay,
      trainable: false
  })

  return batchNormLayer.apply(inputs);

    // return tf2.conv2d(inputs, this.kernal_shape, this.channel_size, 1, this.name);
  }
}
