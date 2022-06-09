import * as tf2 from '@tensorflow/tfjs';
import * as _ from 'underscore';
import '@tensorflow/tfjs-node';
tf2.setBackend('tensorflow');
import * as utils from './utils.js';
import { Episgt } from './Episgt.js';
import { DCModel, DCModelOntar } from './DCModel.js';

export function buildOnTargetModel(inputs) {
    let channelSize = [8, 32, 64, 64, 256, 256];
    let betas = [null]
    for (let i = 1; i < channelSize.length; i++) {
        betas.push(tf2.variable(tf2.tensor(Array(channelSize[i]).fill(0)), 'beta_' + i));
    }

    let e1 = new utils.Conv2D(channelSize[1], [1, 3], 'e1');
    let ebn1u = new utils.BatchNorm(false, 'u1', 0)
    let e2 = new utils.Conv2D(channelSize[2], [1, 3], 'e2', 2);
    let ebn2u = new utils.BatchNorm(false, 'u2', 0)
    let e3 = new utils.Conv2D(channelSize[3], [1, 3], 'e3');
    let ebn3u = new utils.BatchNorm(false, 'u3', 0)
    let e4 = new utils.Conv2D(channelSize[4], [1, 3], 'e4', 2);
    let ebn4u = new utils.BatchNorm(false, 'u4', 0)
    let e5 = new utils.Conv2D(channelSize[5], [1, 3], 'e5');
    let ebn5u = new utils.BatchNorm(false, 'u5', 0)

    let encoder = [null, e1, e2, e3, e4, e5];
    let encoder_u = [null, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u];
    let hu0 = inputs
    let u_lst = hu0
    let hu_lst = hu0

    for (let i = 1; i < channelSize.length; i++) {
        var hu_pre = hu_lst[i - 1]
        var pre_u = encoder[i].conv(hu_pre)
        var u = encoder_u[i].norm(pre_u)
        const hu = tf2.relu(u.add(betas[i]))
        hu_lst.push(hu)
        u_lst.push(u)
    }

    let hu_m1 = hu_lst[channelSize.length - 1]
    let pre_u_last = encoder[channelSize.length - 1].conv(hu_m1)
    let u_last = encoder_u[channelSize.length - 1].norm(pre_u_last)
    u_last = tf2.add(u_last, betas[channelSize.length - 1])
    let hu_last = tf2.relu(u_last)
    hu_lst.push(hu_last)
    u_lst.push(u_last)

    // Classifier
    let cls_channel_size = [512, 512, 1024, 2]
    let e6 = new utils.Conv2D(cls_channel_size[0], [1, 3], 'e6', 2);
    let ebn6u = new utils.BatchNorm(false, 'u6', 0.99)
    let e7 = new utils.Conv2D(cls_channel_size[1], [1, 3], 'e7');
    let ebn7u = new utils.BatchNorm(false, 'u7', 0.99)
    let e8 = new utils.Conv2D(cls_channel_size[2], [1, 3], 'e8', 1, 'valid');
    let ebn8u = new utils.BatchNorm(false, 'u8', 0.99)
    let e9 = new utils.Conv2D(cls_channel_size[3], [1, 1], 'e9');
    let ebn9u = new utils.BatchNorm(false, 'u9', 0.99)

    let cls_layers = [null, e6, e7, e8, e9];
    let cls_bn_layers = [null, ebn6u, ebn7u, ebn8u, ebn9u];
    let hl0 = hu_last
    let l_lst = [hl0]
    let hl_lst = [hl0]

    for (let i = 1; i <= cls_channel_size.length; i++) {
        var hl_pre = hl_lst[i - 1]
        var pre_l = cls_layers[i].conv(hl_pre)
        var l = cls_bn_layers[i].norm(pre_l)
        const hl = tf2.relu(l)
        hl_lst.push(hl)
        l_lst.push(l)
    }

    let hl_m1 = hl_lst[cls_channel_size.length - 1]
    let l_last = cls_layers[cls_channel_size.length - 1].conv(hl_m1)
    let hl_last = tf2.softmax(l_last)
    hl_lst.push(hl_last)
    l_lst.push(l_last)

    let sig_l = tf2.squeeze(hl_last)
    return sig_l
}




(async function () {
const a = new Episgt('examples/eg_cls_on_target.episgt', 4, true)

let results = await a.get_dataset()
let x = results[0]
let y = results[1]
x = tf2.expandDims(x, 2)
let training = new DCModel().train(x, y)
// console.log(x , y)
})();