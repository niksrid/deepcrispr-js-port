import * as tf2 from '@tensorflow/tfjs';
import { buildOnTargetModel } from './run_examples.js'
import * as dfd from 'danfojs-node';
import { noop } from 'underscore';
import { sum } from '@tensorflow/tfjs';

export class DCModel {
  constructor(ontar_model_dir, is_reg = false, seq_feature_only = false) {
    this.ontar_model_dir = ontar_model_dir;
    this.is_reg = is_reg;
    this.seq_feature_only = seq_feature_only;
  }

  train(x, y, batch_size = 500, epochs = 50) {
    let inputs_sg = [tf2.tensor([], [0, 1, 23, 8], 'float32')]
    console.log("training begun")
    let pred_ontar = buildOnTargetModel(inputs_sg)
    let dataset = new Dataset(x, y)
    let optimizer = tf2.train.adam(0.001)
    for (let i = 0; i < epochs; i++) {
        let [data_input, labels] = dataset.next_batch(batch_size);
        labels = labels.variable(true)
        // tf2.losses.meanSquaredError(labels, pred_ontar)
      
        data_input = data_input.transpose([0, 2, 3, 1]).variable(true)
        let pred_ontar = buildOnTargetModel([data_input])
        pred_ontar = pred_ontar.variable(true)
        
       let loss_train = optimizer.minimize(()=>{
          return tf2.losses.softmaxCrossEntropy(tf2.oneHot(tf2.oneHot(labels.asType('int32'), 4), 2), pred_ontar, true)
        }, true)

        this.epochs_completed += 1;
        if (i % 10 == 0) {
          console.log("Epoch:", i + 1)
          console.log('Loss:', loss_train.dataSync())
        }
    }
    x = x.transpose([0, 2, 3, 1])
    let yp = buildOnTargetModel([x])
    let preNorm = new dfd.Series(y).valueCounts()
    let encode = new dfd.LabelEncoder()
    encode.fit(preNorm.values)
    console.log("Results if you just guessed:")
    console.log(encode.transform(preNorm.values), preNorm.values)
    console.log("Our results:")
    console.log(yp.dataSync().map(Math.round).filter(x=>x===1).reduce((a,c)=>a+c, 0) / yp.size)

  }

  ontar_predict(x, channel_first = true) {
    if (channel_first) {
      x = x.transpose([0, 2, 3, 1])
    }
    let yp = buildOnTargetModel([x])
    return yp.dataSync()
  
  }
}

export class DCModelOntar {
  constructor(ontar_model_dir, is_reg = false, seq_feature_only = false) {
    this.ontar_model_dir = ontar_model_dir;
    this.is_reg = is_reg;
    this.seq_feature_only = seq_feature_only;
    this.pred_ontar = buildOnTargetModel(this.inputs_sg)
  }


}


export class Dataset {
  constructor(data, labels) {
    this.data = data;
    this.labels = labels;
    this.epochs_completed = 0;
    this.index_in_epoch = 0;
    this.num_examples = labels.shape[0];
  }

  next_batch(batch_size) {
    let n_sample = this.num_examples;
    let start = this.index_in_epoch;
    let end = this.index_in_epoch + batch_size;
    end = Math.min(end, n_sample);
    let id = tf2.range(start, end);
    let data_input = this.data.gather(id.asType('int32'));
    let labels = this.labels.gather(id.asType('int32'));
    
    this.index_in_epoch = end;


    if (end == n_sample) {
      this.epochs_completed += 1;
      this.index_in_epoch = 0;
    }

    return [data_input, labels];

  }

}