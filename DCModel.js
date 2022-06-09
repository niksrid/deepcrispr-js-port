import * as tf2 from '@tensorflow/tfjs';
import {buildOnTargetModel} from './run_examples.js'
import * as dfd from 'danfojs-node';
import { noop } from 'underscore';

export class DCModel {
  constructor(ontar_model_dir, is_reg = false, seq_feature_only=false){
    this.ontar_model_dir = ontar_model_dir;
    this.is_reg = is_reg;
    this.seq_feature_only = seq_feature_only;
  }



  train(x, y, batch_size=500, epochs=50){
   let inputs_sg = [tf2.tensor([], [0, 1, 23, 8], 'float32')]
    console.log("training begun")
   let pred_ontar = buildOnTargetModel(inputs_sg)
   let labels_pl = tf2.tensor([], [0, 8, 1024], 'float32').variable()
    let dataset = new Dataset(x, y)
    let optimizer = tf2.train.adam(0.001)
    let train_op = optimizer.minimize(() => tf2.losses.softmaxCrossEntropy(labels_pl, pred_ontar))
    for (let i = 0; i < epochs; i++) {
      for (let j = 0; j < dataset.num_examples; j += batch_size) {
        console.log(batch_size)
        let [data_input, labels] = dataset.next_batch(batch_size);
        labels_pl.assign(labels);
        train_op.run({inputs_sg, labels_pl});
      }
      this.epochs_completed += 1;
    }
    
    yp = buildOnTargetModel(x)
    let preNorm = new dfd.Series(y).valueCounts()
    let encode = new dfd.LabelEncoder()

    console.log("Results if you just guessed:")
    console.log(encode.transform(preNorm.values))
    console.log("Our results:")
    console.log(Math.round(yp)/y.length)

  }
}

export class DCModelOntar {
constructor(ontar_model_dir, is_reg = false, seq_feature_only=false){
    this.ontar_model_dir = ontar_model_dir;
    this.is_reg = is_reg;
    this.seq_feature_only = seq_feature_only;
    this.pred_ontar = buildOnTargetModel(this.inputs_sg)
  }

  ontar_predict(x, channel_first=true){
    if(channel_first){
      x = x.transpose([0, 2, 3, 1])
    }
    return this.pred_ontar.predict(fd)
  }
}


export class Dataset {
  constructor(data, labels){
    this.data = data;
    this.labels = labels;
    this.epochs_completed = 0;
    this.index_in_epoch = 0;
    this.num_examples = labels.shape[0];
  } 

  next_batch(batch_size){
    let n_sample = this.num_examples;
    let start = this.index_in_epoch;
    let end = this.index_in_epoch + batch_size;
     end = Math.min(end, n_sample);
    let id = tf2.range(start, end);
    this.index_in_epoch = end;
   let data_input = tf2.gather(this.data, id.asType('int32'));
    let labels = tf2.gather(this.labels, id.asType('int32'));
    
    return [data_input, labels];

  }

}