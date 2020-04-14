"""Visual Attention Based OCR Model."""

from __future__ import absolute_import
from __future__ import division

import time
import os
import math
import logging
import sys
from functools import reduce

import distance
import numpy as np
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin
from .cnn import CNN
from .seq2seq_model import Seq2SeqModel
from ..util.data_gen import DataGen
from ..util.visualizations import visualize_attention


class Model(object):
    def __init__(self,
                 phase,
                 visualize,
                 output_dir,
                 batch_size,
                 initial_learning_rate,
                 steps_per_checkpoint,
                 model_dir,
                 target_embedding_size,
                 attn_num_hidden,
                 attn_num_layers,
                 clip_gradients,
                 max_gradient_norm,
                 session,
                 load_model,
                 gpu_id,
                 use_gru,
                 use_distance=True,
                 max_image_width=160,
                 max_image_height=60,
                 max_prediction_length=8,
                 channels=1,
                 reg_val=0):

        self.use_distance = use_distance

        self.max_original_width = max_image_width
        self.max_original_height = max_image_height

        # We need resized width, not the actual width
        self.height = DataGen.IMAGE_HEIGHT
        max_resized_width = 1. * max_image_width / max_image_height * self.height
        self.max_width = int(math.ceil(max_resized_width))

        self.encoder_size = int(math.ceil(1. * self.max_width / 4))
        self.decoder_size = max_prediction_length + 2
        self.buckets = [(self.encoder_size, self.decoder_size)]

        if gpu_id >= 0:
            device_id = '/gpu:' + str(gpu_id)
        else:
            device_id = '/cpu:0'
        self.device_id = device_id

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if phase == 'test':
            batch_size = 1

        logging.info('phase: %s', phase)
        logging.info('model_dir: %s', model_dir)
        logging.info('load_model: %s', load_model)
        logging.info('output_dir: %s', output_dir)
        logging.info('steps_per_checkpoint: %d', steps_per_checkpoint)
        logging.info('batch_size: %d', batch_size)
        logging.info('learning_rate: %f', initial_learning_rate)
        logging.info('reg_val: %d', reg_val)
        logging.info('max_gradient_norm: %f', max_gradient_norm)
        logging.info('clip_gradients: %s', clip_gradients)
        logging.info('max_image_width %f', max_image_width)
        logging.info('max_prediction_length %f', max_prediction_length)
        logging.info('channels: %d', channels)
        logging.info('target_embedding_size: %f', target_embedding_size)
        logging.info('attn_num_hidden: %d', attn_num_hidden)
        logging.info('attn_num_layers: %d', attn_num_layers)
        logging.info('visualize: %s', visualize)

        if use_gru:
            logging.info('using GRU in the decoder.')

        self.reg_val = reg_val
        self.sess = session
        self.steps_per_checkpoint = steps_per_checkpoint
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.phase = phase
        self.visualize = visualize
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients
        self.channels = channels

        if phase == 'train':
            self.forward_only = False
        else:
            self.forward_only = True

        with tf.device(device_id):

            self.img_pl = tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=(None, DataGen.IMAGE_HEIGHT, self.max_width, self.channels),
                name='input')

            num_images = tf.shape(input=self.img_pl)[0]

            # TODO: create a mask depending on the image/batch size
            self.encoder_masks = []
            for i in xrange(self.encoder_size + 1):
                self.encoder_masks.append(
                    tf.tile([[1.]], [num_images, 1])
                )

            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(self.decoder_size + 1):
                self.decoder_inputs.append(
                    tf.tile([1], [num_images])
                )
                if i < self.decoder_size:
                    self.target_weights.append(tf.tile([1.], [num_images]))
                else:
                    self.target_weights.append(tf.tile([0.], [num_images]))

            cnn_model = CNN(self.img_pl, not self.forward_only)
            self.conv_output = cnn_model.tf_output()
            self.perm_conv_output = tf.transpose(a=self.conv_output, perm=[1, 0, 2])
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks=self.encoder_masks,
                encoder_inputs_tensor=self.perm_conv_output,
                decoder_inputs=self.decoder_inputs,
                target_weights=self.target_weights,
                target_vocab_size=len(DataGen.CHARMAP),
                buckets=self.buckets,
                target_embedding_size=target_embedding_size,
                attn_num_layers=attn_num_layers,
                attn_num_hidden=attn_num_hidden,
                forward_only=self.forward_only,
                use_gru=use_gru)

            num_feed = []
            prb_feed = []

            for line in xrange(len(self.attention_decoder_model.output)):
                guess = tf.argmax(input=self.attention_decoder_model.output[line], axis=1)
                proba = tf.reduce_max(
                    input_tensor=tf.nn.softmax(self.attention_decoder_model.output[line]), axis=1)
                num_feed.append(guess)
                prb_feed.append(proba)

            # In the original code the predictions are converted into
            # the corresponding characters using a mutable hash table.
            # Then, they are joined into a single string. Since the
            # mutable hash table does not exist in the 2.0 API covered
            # with the fact that it's a bit of a hustle to use control
            # flow operations in tf lite, the string will be obtained
            # outside the model based on the same charmap used in the
            # training process.
            trans_output = tf.transpose(a=num_feed)

            # Again, in the original code the probability is calculated
            # using the map_fn functionality, but that causes an error
            # in the tf lite conversion stage so it'll not be used here.
            # Instead, the total probability will be obtained outside
            # the model.
            trans_outprb = tf.transpose(a=prb_feed)

            self.prediction = tf.identity(trans_output, name='prediction')
            self.probability = tf.identity(trans_outprb, name='probability')

            if not self.forward_only:  # train
                self.updates = []
                self.summaries_by_bucket = []

                params = tf.compat.v1.trainable_variables()
                opt = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)
                loss_op = self.attention_decoder_model.loss

                if self.reg_val > 0:
                    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
                    logging.info('Adding %s regularization losses', len(reg_losses))
                    logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
                    loss_op = self.reg_val * tf.reduce_sum(input_tensor=reg_losses) + loss_op

                gradients, params = list(zip(*opt.compute_gradients(loss_op, params)))
                if self.clip_gradients:
                    gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

                # Summaries for loss, variables, gradients, gradient norms and total gradient norm.
                summaries = [
                    tf.compat.v1.summary.scalar("loss", loss_op),
                    tf.compat.v1.summary.scalar("total_gradient_norm", tf.linalg.global_norm(gradients))
                ]
                all_summaries = tf.compat.v1.summary.merge(summaries)
                self.summaries_by_bucket.append(all_summaries)

                # update op - apply gradients
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.updates.append(
                        opt.apply_gradients(
                            list(zip(gradients, params)),
                            global_step=self.global_step
                        )
                    )

        self.saver_all = tf.compat.v1.train.Saver(tf.compat.v1.all_variables())
        self.checkpoint_path = os.path.join(self.model_dir, "model.ckpt")

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and load_model:
            # pylint: disable=no-member
            logging.info("Reading model parameters from %s", ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            self.sess.run(tf.compat.v1.initialize_all_variables())

    def predict(self, image_file_data):
        input_feed = {}
        input_feed[self.img_pl.name] = image_file_data

        output_feed = [self.prediction, self.probability]
        outputs = self.sess.run(output_feed, input_feed)

        text = outputs[0]
        probability = outputs[1]
        if sys.version_info >= (3,):
            text = text.decode('iso-8859-1')

        return (text, probability)

    def test(self, data_path):

        def get_word_from_sequence(sequence):
            return ''.join(DataGen.CHARMAP[j] for j in sequence)

        def get_total_probability(probabilities):
            return reduce(lambda x, y: x * y, probabilities)

        current_step = 0
        num_correct = 0.0
        num_total = 0.0

        s_gen = DataGen(data_path, self.buckets, epochs=1, max_width=self.max_original_width,
                        max_height=self.max_original_height, channels=self.channels)
        for batch in s_gen.gen(1):
            current_step += 1
            # Get a batch (one image) and make a step.
            start_time = time.time()
            result = self.step(batch, self.forward_only)
            curr_step_time = (time.time() - start_time)

            num_total += 1

            output = result['prediction']
            # you want to decode the output as it's no longer done in the model graph
            output = get_word_from_sequence(output[0])
            ground = batch['labels'][0]
            comment = batch['comments'][0]
            if sys.version_info >= (3,):
                ground = ground.decode('iso-8859-1')
                comment = comment.decode('iso-8859-1')

            probability = result['probability']

            # you also want to get the total probability as it's no longer done in the model graph
            probability = get_total_probability(probability[0])

            if self.use_distance:
                incorrect = distance.levenshtein(output, ground)
                if not ground:
                    if not output:
                        incorrect = 0
                    else:
                        incorrect = 1
                else:
                    incorrect = float(incorrect) / len(ground)
                incorrect = min(1, incorrect)
            else:
                incorrect = 0 if output == ground else 1

            num_correct += 1. - incorrect

            if self.visualize:
                # Attention visualization.
                threshold = 0.5
                normalize = True
                binarize = True
                attns_list = [[a.tolist() for a in step_attn] for step_attn in result['attentions']]
                attns = np.array(attns_list).transpose([1, 0, 2])
                visualize_attention(batch['data'],
                                    'out',
                                    attns,
                                    output,
                                    self.max_width,
                                    self.height,
                                    threshold=threshold,
                                    normalize=normalize,
                                    binarize=binarize,
                                    ground=ground,
                                    flag=None)

            step_accuracy = "{:>4.0%}".format(1. - incorrect)
            if incorrect:
                correctness = step_accuracy + " ({} vs {}) {}".format(output, ground, comment)
            else:
                correctness = step_accuracy + " (" + ground + ")"

            logging.info('Step {:.0f} ({:.3f}s). '
                         'Accuracy: {:6.2%}, '
                         'loss: {:f}, perplexity: {:0<7.6}, probability: {:6.2%} {}'.format(
                             current_step,
                             curr_step_time,
                             num_correct / num_total,
                             result['loss'],
                             math.exp(result['loss']) if result['loss'] < 300 else float('inf'),
                             probability,
                             correctness))

    def train(self, data_path, num_epoch):
        logging.info('num_epoch: %d', num_epoch)
        s_gen = DataGen(
            data_path, self.buckets,
            epochs=num_epoch,
            max_width=self.max_original_width,
            max_height=self.max_original_height,
            channels=self.channels)
        step_time = 0.0
        loss = 0.0
        current_step = 0
        skipped_counter = 0
        writer = tf.compat.v1.summary.FileWriter(self.model_dir, self.sess.graph)

        logging.info('Starting the training process.')
        for batch in s_gen.gen(self.batch_size):

            current_step += 1

            start_time = time.time()
            # result = self.step(batch, self.forward_only)
            result = None
            try:
                result = self.step(batch, self.forward_only)
            except Exception as e:
                skipped_counter += 1
                logging.info("Step {} failed, batch skipped." +
                             " Total skipped: {}".format(current_step, skipped_counter))
                logging.error(
                    "Step {} failed. Exception details: {}".format(current_step, str(e)))
                continue

            loss += result['loss'] / self.steps_per_checkpoint
            curr_step_time = (time.time() - start_time)
            step_time += curr_step_time / self.steps_per_checkpoint

            # num_correct = 0

            # step_outputs = result['prediction']
            # grounds = batch['labels']
            # for output, ground in zip(step_outputs, grounds):
            #     if self.use_distance:
            #         incorrect = distance.levenshtein(output, ground)
            #         incorrect = float(incorrect) / len(ground)
            #         incorrect = min(1.0, incorrect)
            #     else:
            #         incorrect = 0 if output == ground else 1
            #     num_correct += 1. - incorrect

            writer.add_summary(result['summaries'], current_step)

            # precision = num_correct / len(batch['labels'])
            step_perplexity = math.exp(result['loss']) if result['loss'] < 300 else float('inf')

            # logging.info('Step %i: %.3fs, precision: %.2f, loss: %f, perplexity: %f.'
            #              % (current_step, curr_step_time, precision*100,
            #                 result['loss'], step_perplexity))

            logging.info('Step %i: %.3fs, loss: %f, perplexity: %f.',
                         current_step, curr_step_time, result['loss'], step_perplexity)

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % self.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                # Print statistics for the previous epoch.
                logging.info("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f.",
                             self.sess.run(self.global_step), step_time, loss, perplexity)
                # Save checkpoint and reset timer and loss.
                logging.info("Saving the model at step %d.", current_step)
                self.saver_all.save(self.sess, self.checkpoint_path, global_step=self.global_step)
                step_time, loss = 0.0, 0.0

        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        logging.info("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f.",
                     self.sess.run(self.global_step), step_time, loss, perplexity)

        if skipped_counter:
            logging.info("Skipped {} batches due to errors.".format(skipped_counter))

        # Save checkpoint and reset timer and loss.
        logging.info("Finishing the training and saving the model at step %d.", current_step)
        self.saver_all.save(self.sess, self.checkpoint_path, global_step=self.global_step)

    # step, read one batch, generate gradients
    def step(self, batch, forward_only):
        img_data = batch['data']
        decoder_inputs = batch['decoder_inputs']
        target_weights = batch['target_weights']

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_pl.name] = img_data

        for idx in xrange(self.decoder_size):
            input_feed[self.decoder_inputs[idx].name] = decoder_inputs[idx]
            input_feed[self.target_weights[idx].name] = target_weights[idx]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[self.decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        output_feed = [
            self.attention_decoder_model.loss,  # Loss for this batch.
        ]

        if not forward_only:
            output_feed += [self.summaries_by_bucket[0],
                            self.updates[0]]
        else:
            output_feed += [self.prediction]
            output_feed += [self.probability]
            if self.visualize:
                output_feed += self.attention_decoder_model.attentions

        outputs = self.sess.run(output_feed, input_feed)

        res = {
            'loss': outputs[0],
        }

        if not forward_only:
            res['summaries'] = outputs[1]
        else:
            res['prediction'] = outputs[1]
            res['probability'] = outputs[2]
            if self.visualize:
                res['attentions'] = outputs[3:]

        return res
