## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

##### 莱维飞行(10) +　QPSO(50) + 莱维飞行

from deepspeech import Model
import random
import math
import numpy as np
import tensorflow as tf
import argparse
import scipy.io.wavfile as wav
import os
import sys
import time

sys.path.append("DeepSpeech")

import DeepSpeech

# 5.13 跑 python attack_pso.py --input sample_input.wav --target "and you know" --population 50  47.64s

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits
from scipy.signal import butter, lfilter

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"


class Attack:
    def __init__(self, sess, phrase_length, max_audio_len, batch_size=1,
                 restore_path=None):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """

        self.sess = sess
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        # self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                               name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        # Attack里对input添加干扰mask
        self.new_input = new_input = mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape, stddev=2)
        pass_in = tf.clip_by_value(new_input + noise, -2 ** 15, 2 ** 15 - 1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)

        ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                 inputs=logits, sequence_length=lengths)

        self.expanded_loss = tf.constant(0)
        self.ctcloss = ctcloss

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

    def attack(self, audio, lengths, target, print_toggle):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        # sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths) - 1) // 320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(
            np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths) - 1) // 320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t) + [0] * (self.phrase_length - len(t)) for t in target])))

        # We'll make a bunch of iterations of gradient descent here
        el, cl, logits, new_input = sess.run((self.expanded_loss, self.ctcloss, self.logits, self.new_input))

        if print_toggle:
            optimal_index = np.where(cl == min(cl))
            optimal_cost = cl[optimal_index]
            optimal_audio = new_input[optimal_index]
            # Try to retrieve the decoded words
            out = sess.run(self.decoded)

            res = np.zeros(out[0].dense_shape) + len(toks) - 1

            for ii in range(len(out[0].values)):
                x, y = out[0].indices[ii]
                res[x, y] = out[0].values[ii]

            res = ["".join(toks[int(x)] for x in y) for y in res]
            print("======================================================")
            print("Current decoded word (without language model): " + str(res[optimal_index[0][0]]))
            # np.mean(a)求a的平均值
            print("Average loss: %.3f" % np.mean(cl) + "\n")

        # return new audios and new cost
        return new_input, cl


def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)


def mutate_audio(audio, num, mutation_range, elite_pop):
    audios = []
    lengths = []
    noise = np.random.randn(*audio.shape) * 40
    noise = highpass_filter(noise)
    for i in range(num):
        #         wn = np.random.randint(-mutation_range, mutation_range, size=len(audio), dtype=np.int16)
        #         mutated_audio = audio + wn
        mask = np.random.rand(audio.shape[0], elite_pop.shape[1]) < 0.005
        mutated_audio = audio + mask * noise
        audios.append(list(mutated_audio))
        lengths.append(len(mutated_audio))
    return audios, lengths


# 计算编辑距离 --- dist
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)

    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# 计算分贝-----信噪比SNR
def db(audio):
    if len(audio.shape) > 1:
        maxx = np.max(np.abs(audio), axis=1)
        # np.any()中有一个true 就是true
        return 20 * np.log10(maxx) if np.any(maxx != 0) else np.array([0])
    maxx = np.max(np.abs(audio))
    return 20 * np.log10(maxx) if maxx != 0 else np.array([0])


# Create the particle swarming class
class PSOEnvironment():
    '''
    The hyperparameters set in this init function such as w, c1 and c2
    are predetermined from other research, which conclude that these
    values are optimal.
    '''

    def __init__(self, num_particle, audio, model_path, target, iterations, sess):

        self.global_min_cost = float('inf')
        self.gbest_position = audio
        self.target = target
        self.iterations = iterations
        self.lengths = []
        self.ds = self.build_model(model_path)
        self.pop_size = num_particle

        # 梯度估计
        self.num_points_estimate = 100
        self.delta_for_gradient = 100
        self.elite_size = 10
        self.funcs = self.setup_graph(self.pop, np.array([toks.index(x) for x in target_phrase]))

        restore_path = model_path + "/model.v0.4.1"

        audios = []

        # To create the first set of particles here
        # Creating first set of mutation
        for _ in range(num_particle):
            wn = np.random.randint(-200, 200, size=len(audio), dtype=np.int16)
            mutated_audio = audio + wn
            audios.append(list(mutated_audio))
            self.lengths.append(len(mutated_audio))

        maxlen = max(map(len, audios))
        audios = np.array([x + [0] * (maxlen - len(x)) for x in audios])

        self.attack = Attack(sess, len(target), maxlen, batch_size=len(audios), restore_path=restore_path)
        new_input, cl = self.attack.attack(audios, self.lengths, [[toks.index(x) for x in self.target]] * len(audios),
                                           True)
        self.elite_pop = new_input
        # Instantiating the particles
        self.particles = []
        for i in range(num_particle):
            velocity = new_input[i] - audio
            self.particles.append(Particle(new_input[i], cl[i], velocity))

        self.global_min_cost = min(cl)
        # np.where()以元组的形式输出满足条件元素的坐标
        optimal_index = np.where(cl == min(cl))
        self.gbest_position = new_input[optimal_index][0]
        print("================================================")
        print("global best position: " + str(self.gbest_position))

    def setup_graph(self, input_audio_batch, target_phrase):
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)

        pass_in = np.clip(input_audio_batch, -2 ** 15, 2 ** 15 - 1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(tf.float32, shape=pass_in.shape, name='a')
            len_batch = tf.placeholder(tf.float32, name='b')
            arg2_logits = tf.placeholder(tf.int32, shape=logits_arg2.shape, name='c')
            arg1_dense = tf.placeholder(tf.float32, shape=dense_arg1.shape, name='d')
            arg2_dense = tf.placeholder(tf.int32, shape=dense_arg2.shape, name='e')
            len_seq = tf.placeholder(tf.int32, shape=seq_len.shape, name='f')

            logits = get_logits(inputs, arg2_logits)
            target = ctc_label_dense_to_sparse(arg1_dense, arg2_dense, len_batch)
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=len_seq)
            decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg2_logits, merge_repeated=True)

            sess = tf.Session()
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, restore_path)

        func1 = lambda a, b, c, d, e, f: sess.run(ctcloss,
                                                  feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d,
                                                             arg2_dense: e, len_seq: f})
        func2 = lambda a, b, c, d, e, f: sess.run([ctcloss, decoded],
                                                  feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d,
                                                             arg2_dense: e, len_seq: f})
        return (func1, func2)

    def print_positions(self):
        for particle in self.particles:
            particle.__str__()

    def print_best_audio(self, audio, log=None):
        if log is not None:
            log.write('target phrase: ' + self.target + '\n')
            log.write('itr, loss, dist, corr, db, decoded \n')
        for i in range(self.iterations):
            print_toggle = False
            print("Iteration: " + str(i))
            if (i + 1) % 10 == 0:
                print_toggle = True

            decoded = self.ds.stt(self.gbest_position.astype(np.int16), 16000)
            corr = "{0:.4f}".format(np.corrcoef(audio, self.gbest_position)[0][1])  # 皮尔逊相关系数
            dist = levenshteinDistance(decoded, self.target)

            # Update the particle position
            self.update(print_toggle, i, audio, dist)

            print(
                "Current decoded word: " + decoded + "\t" + "Cost: " + str(
                    self.global_min_cost))
            print('output dB', db(self.gbest_position))  # 信噪比SNR
            print('dist', dist)  # 编辑距离

            if log is not None:
                log.write(str(i) + ", " + str(self.global_min_cost) + ", " + str(dist) + ", " + str(
                    corr) + ", " + str(db(self.gbest_position)) + ", " + decoded + "\n")
            if (decoded == self.target):
                return True
                break
        # Save and output the audio file
        out_wav_file = 'levy_qpso_adv.wav'
        wav.write(out_wav_file, 16000, self.gbest_position.astype(np.int16))

    def levy(self, audios):
        # Levy flights by Mantegna 's algorithm
        beta = 1.5
        alpha = 1
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                beta * math.gamma((1 + beta) / 2) * (2 ** ((beta - 1) / 2)))) ** (
                          1 / beta)
        sigma_v = 1

        for particle in self.particles:
            u = np.random.normal(0, sigma_u, 1)
            v = np.random.normal(0, sigma_v, 1)
            step = u / ((abs(v)) ** (1 / beta))
            # lamuda的设置关系到点的活力程度，方向由pbest决定，有点类似PSO，但步长不一样
            lamuda = 1
            stepsize = lamuda * step * (particle.position - self.gbest_position)
            particle.position = particle.position + stepsize * np.random.randn(
                len(particle.position), )  # 产生满足正态分布的序列
            particle.move_particle()
            audios.append(particle.position)
        return audios

    def QPSO(self, t, audios):
        # QPSO algorithm
        # 计算mbest
        global sum_mbest
        sum_mbest = np.zeros((len(self.gbest_position),))
        for particle in self.particles:
            sum_mbest += particle.pbest_position
        mbest = (sum_mbest) / self.pop_size

        for particle in self.particles:
            # 粒子位置更新
            a = random.uniform(0, 1)
            pi = a * particle.pbest_position + (1 - a) * self.gbest_position
            u = random.uniform(0, 1)
            B = 0.5 * (self.iterations - t) / self.iterations + 0.5

            if random.random() > 0.5:
                particle.position = pi + math.log(1 / u) * B * abs(np.array(mbest - particle.position))
            else:
                particle.position = pi - math.log(1 / u) * B * abs(np.array(mbest - particle.position))
            particle.move_particle()
            audios.append(particle.position)
        return audios

    def getctcloss(self, input_audio_batch, target_phrase, decode=False):
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)

        pass_in = np.clip(input_audio_batch, -2 ** 15, 2 ** 15 - 1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)

        if decode:
            return self.funcs[1](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
        else:
            return self.funcs[0](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)

    def get_fitness_score(self, input_audio_batch, target_phrase, input_audio, classify=False):
        target_enc = np.array([toks.index(x) for x in target_phrase])
        if classify:
            ctcloss, decoded = self.getctcloss(input_audio_batch, target_enc, decode=True)
            all_text = "".join([toks[x] for x in decoded[0].values])
            index = len(all_text) // input_audio_batch.shape[0]
            final_text = all_text[:index]
        else:
            ctcloss = self.getctcloss(input_audio_batch, target_enc)
        score = -ctcloss
        if classify:
            return (score, final_text)
        return score, -ctcloss

    def update(self, print_toggle, t, audio, dist):
        audios = []
        mutation_population = self.pop_size
        if t < 10:
            self.levy(audios)
        elif t > 10 & dist > 2:
            self.QPSO(t, audios)
            flag = "true"
            # # 变异操作
            # mutated_audios, mutated_lengths = mutate_audio(audio, mutation_population, 150, self.elite_pop)
            #
            # audios.extend(mutated_audios)

        else:
            pop_scores, ctc = self.get_fitness_score(self.particles, self.target, audio)
            elite_ind = np.argsort(pop_scores)[-self.elite_size:]
            elite_pop, elite_pop_scores, elite_ctc = self.pop[elite_ind], pop_scores[elite_ind], ctc[elite_ind]

            perturbed = np.tile(np.expand_dims(elite_pop[-1], axis=0), (self.num_points_estimate, 1))
            indices = np.random.choice(audios.shape[1], size=self.num_points_estimate, replace=False)

            perturbed[np.arange(self.num_points_estimate), indices] += self.delta_for_gradient
            perturbed_scores = self.get_fitness_score(perturbed, self.target, self.audio)[0]

            grad = (perturbed_scores - elite_ctc[-1]) / self.delta_for_gradient
            grad /= np.abs(grad).max()
            modified = elite_pop[-1].copy()
            modified[indices] += grad * self.delta_for_perturbation

            self.particles = np.tile(np.expand_dims(modified, axis=0), (self.pop_size, 1))
            self.delta_for_perturbation *= 0.995

        # calculate new cost
        # 迭代10次 print_toggle为true 为true之后计算Current decoded word (without language model):以及Average loss: %.3f"
        new_input, cl = self.attack.attack(audios, self.lengths,
                                           [[toks.index(x) for x in self.target]] * len(audios),
                                           print_toggle)
        # update my particles
        for i, particle in enumerate(self.particles):
            # 极值更新
            if cl[i] < particle.min_cost:
                particle.min_cost = cl[i]
                particle.pbest_position = new_input[i]

            if cl[i] < self.global_min_cost:
                self.gbest_position = new_input[i]
                self.global_min_cost = cl[i]

    #
    def build_model(self, model_path):

        # Build deepspeech model to use for adversarial sample evaluation
        BEAM_WIDTH = 500
        LM_ALPHA = 0.75
        LM_BETA = 1.85
        N_FEATURES = 26
        N_CONTEXT = 9
        MODEL_PATH = model_path + '/models/output_graph.pb'
        ALPHABET_PATH = model_path + '/models/alphabet.txt'
        LM_PATH = model_path + '/models/lm.binary'
        TRIE_PATH = model_path + '/models/trie'

        ds = Model(MODEL_PATH, N_FEATURES, N_CONTEXT, ALPHABET_PATH, BEAM_WIDTH)
        ds.enableDecoderWithLM(ALPHABET_PATH, LM_PATH, TRIE_PATH, LM_ALPHA, LM_BETA)

        return ds


'''
Particle Class is used to track the individual particle's position,
velocity and personal best position. Instantiated with the same 3 
variables. 
'''


class Particle():
    def __init__(self, position, cost, velocity):
        self.position = position
        self.min_cost = cost
        self.pbest_position = position
        self.velocity = velocity

    '''
    Called by the update function in PSOEnvironment to update the 
    position of each particle.
    '''

    def __str__(self):
        print("Position: " + str(self.position) + "\nCost: " + str(self.min_cost))


# This is the main function. None of the algorithm takes place here.
# Starts by taking in commandline arguments
# Instantiate the PSOEnvironment class
# Iterate the attack using the PSOEnvironment object
def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--target', type=str,
                        required=True,
                        help="Target transcription")

    parser.add_argument('--population', type=int,
                        required=False, default=100,
                        help="Population size of each generation")

    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    population_size = args.population
    model_path = "deepspeech-0.4.1-checkpoint"
    iterations = 1000
    target = args.target

    with tf.Session() as sess:
        audios = []
        lengths = []

        # Load the inputs that we're given
        fs, audio = wav.read(args.input)
        assert fs == 16000
        assert audio.dtype == np.int16
        out_wav_file = 'levy_qpso_adv.wav'
        log_file = 'levy_qpso_log.txt'
        # Instantiate the PSOEnvironment object
        start = time.time()
        pso_environment = PSOEnvironment(population_size, audio, model_path, target, iterations, sess)
        with open(log_file, 'w') as log:
            success = pso_environment.print_best_audio(audio, log=log)

        if success:
            print('Succes! Wav file stored as', out_wav_file)
            end = time.time()
            print("adversarial example is successfully generated with {} s".format(end - start))
        else:
            print('Not totally a success! Consider running for more iterations. Intermediate output stored as',
                  out_wav_file)


main()
