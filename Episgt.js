import * as pd from 'pandas-js';
import * as tf2 from '@tensorflow/tfjs';
import { reduce } from 'underscore';

import * as dfd from 'danfojs-node';

const ntmap = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}
const epimap = { 'A': 1, 'N': 0 }

export class Episgt {
    constructor(fpath, num_epi_features, with_y) {
        this.init = new Promise((resolve, reject) => {
            dfd.readCSV(fpath, {
                header: false,
                delimiter: '\t',
            }).then((df) => {
                this.fpath = fpath
                this.num_epi_features = num_epi_features
                this.with_y = with_y || true
                this.ori_df = df
                this.num_cols = with_y ? num_epi_features + 2 : num_epi_features + 1
                this.cols = (this.ori_df.columns)
                .slice(this.ori_df.columns.length - this.num_cols)
                this.df = this.ori_df.copy().drop({
                    columns: this.ori_df.copy().columns.slice(0,this.ori_df.columns.length - this.num_cols)
                })

                resolve(this)
            })
                .catch((err) => {
                    reject(err)
                }
                )
        })
    }

    get_dataset(x_dtype = 'float32', y_dtype = 'float32') {
        return this.init.then((self) => {
        return new Promise((resolve, reject) => {
        let x_seq = tf2.concat([this.df.values.map(x => this.get_seqcode(x[0]))].flat())
        let x_epi = tf2.concat(Array(this.num_epi_features).fill(0).map((y, i) => {
            return tf2.concat(this.df.values.map(x => this.get_epicode(x[i+1])))
        }), -1)

        // x_epi = x_epi.reshape([100, 23, 4])
        let x = tf2.concat([x_seq, x_epi], -1).asType(x_dtype)
        x = x.transpose([0, 2, 1])
        if (this.with_y) {
          let  y = (this.ori_df.copy().drop({
            columns:this.ori_df.copy().columns.slice(0,this.ori_df.columns.length - 1)
        }))
        y = tf2.tensor(y.values.flat())
            return resolve([x, y])
        }
        else {
            return resolve(x)
        }
    })
    })
    }

    get_seqcode(seq) {
        return tf2.tensor(seq.split('').map((x) => {
            return (ntmap[x.toUpperCase()])
        })).reshape([1, seq.length, -1])
    }

    get_epicode(epi) {
        return tf2.tensor(epi.split('').map((x) => {
            return [epimap[x.toUpperCase()]]
        })).reshape([1, epi.length, -1])
    }

    add = (a, b) => a + b


}