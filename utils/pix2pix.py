import os, re, cv2, math, numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

def down_block(filters, apply_batchnorm=True):
    block = keras.Sequential()
    block.add(layers.Conv2D(filters, 4, strides=2, padding="same", use_bias=not apply_batchnorm))
    if apply_batchnorm:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())
    return block

def up_block(filters, apply_dropout=False):
    block = keras.Sequential()
    block.add(layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False))
    block.add(layers.BatchNormalization())
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    block.add(layers.ReLU())
    return block

def build_generator(image_size=(256,256), in_channels=1, out_channels=1):
    inputs = layers.Input(shape=(image_size[0], image_size[1], in_channels))
    downs = [
        down_block(64,  apply_batchnorm=False),
        down_block(128),
        down_block(256),
        down_block(512),
        down_block(512),
        down_block(512),
        down_block(512),
        down_block(512),
    ]
    x = inputs
    skips = []
    for d in downs:
        x = d(x); skips.append(x)
    skips = skips[:-1][::-1]

    ups = [
        up_block(512, True),
        up_block(512, True),
        up_block(512, True),
        up_block(512, False),
        up_block(256, False),
        up_block(128, False),
        up_block(64,  False),
    ]
    for up, skip in zip(ups, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    out = layers.Conv2DTranspose(out_channels, 4, strides=2, padding="same", activation="tanh")(x)
    return keras.Model(inputs, out, name="generator")

def build_discriminator(image_size=(256,256), in_channels=1, cond_channels=1):
    inp = layers.Input(shape=(image_size[0], image_size[1], in_channels))   # condition (T2)
    tar = layers.Input(shape=(image_size[0], image_size[1], cond_channels)) # target (CT)
    x = layers.Concatenate()([inp, tar])
    def dlayer(x, f, s=2, bn=True):
        x = layers.Conv2D(f, 4, strides=s, padding="same", use_bias=not bn)(x)
        if bn: x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x
    x = dlayer(x, 64,  bn=False)
    x = dlayer(x, 128)
    x = dlayer(x, 256)
    x = dlayer(x, 512, s=1)
    out = layers.Conv2D(1, 4, strides=1, padding="same")(x)  # logits
    return keras.Model([inp, tar], out, name="discriminator")

class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator, l1_lambda=100.0):
        super().__init__()
        self.gen = generator
        self.disc = discriminator
        self.l1_lambda = l1_lambda
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

        self.m_gen = keras.metrics.Mean(name="gen_total")
        self.m_gan = keras.metrics.Mean(name="gen_gan")
        self.m_l1  = keras.metrics.Mean(name="gen_l1")
        self.m_disc= keras.metrics.Mean(name="disc")
        self.m_psnr= keras.metrics.Mean(name="psnr")
        self.m_ssim= keras.metrics.Mean(name="ssim")

    def compile(self, gen_opt, disc_opt, **kwargs):
        super().compile(**kwargs)
        self.gen_opt, self.disc_opt = gen_opt, disc_opt

    def train_step(self, data):
        x, y = data  # both in [-1,1]
        with tf.GradientTape(persistent=True) as tape:
            fake = self.gen(x, training=True)
            d_real = self.disc([x, y], training=True)
            d_fake = self.disc([x, fake], training=True)

            # losses
            gen_gan = self.bce(tf.ones_like(d_fake), d_fake)
            gen_l1  = tf.reduce_mean(tf.abs(y - fake))
            gen_tot = gen_gan + self.l1_lambda * gen_l1

            d_real_loss = self.bce(tf.ones_like(d_real), d_real)
            d_fake_loss = self.bce(tf.zeros_like(d_fake), d_fake)
            d_loss = 0.5 * (d_real_loss + d_fake_loss)

        g_grads = tape.gradient(gen_tot, self.gen.trainable_variables)
        d_grads = tape.gradient(d_loss, self.disc.trainable_variables)
        del tape
        self.gen_opt.apply_gradients(zip(g_grads, self.gen.trainable_variables))
        self.disc_opt.apply_gradients(zip(d_grads, self.disc.trainable_variables))

        # metrics
        y01, f01 = (y+1)/2, (fake+1)/2
        psnr = tf.reduce_mean(tf.image.psnr(y01, f01, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(y01, f01, max_val=1.0))

        self.m_gen.update_state(gen_tot); self.m_gan.update_state(gen_gan)
        self.m_l1.update_state(gen_l1);   self.m_disc.update_state(d_loss)
        self.m_psnr.update_state(psnr);   self.m_ssim.update_state(ssim)

        return {"gen": self.m_gen.result(), "gen_gan": self.m_gan.result(),
                "gen_l1": self.m_l1.result(), "disc": self.m_disc.result(),
                "psnr": self.m_psnr.result(), "ssim": self.m_ssim.result()}

    def test_step(self, data):
        x, y = data
        fake = self.gen(x, training=False)
        d_real = self.disc([x, y], training=False)
        d_fake = self.disc([x, fake], training=False)

        gen_gan = self.bce(tf.ones_like(d_fake), d_fake)
        gen_l1  = tf.reduce_mean(tf.abs(y - fake))
        gen_tot = gen_gan + self.l1_lambda * gen_l1
        d_loss  = 0.5*(self.bce(tf.ones_like(d_real), d_real) + self.bce(tf.zeros_like(d_fake), d_fake))

        y01, f01 = (y+1)/2, (fake+1)/2
        psnr = tf.reduce_mean(tf.image.psnr(y01, f01, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(y01, f01, max_val=1.0))

        return {"gen": gen_tot, "gen_gan": gen_gan, "gen_l1": gen_l1,
                "disc": d_loss, "psnr": psnr, "ssim": ssim}