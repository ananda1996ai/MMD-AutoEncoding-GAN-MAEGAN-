# -*- coding: utf-8 -*-
## MAEGAN
- MMD Auto Encoding Generative Adversarial Networks
"""

#%tensorflow_version 1.x           #This code is compatible with tensorflow 1.5.x and may not run on Tensorflow 2.x environments.

"""- `lambda_` is a hyperparameter to tune the weightage of the latent loss. 
- `beta_` is a hyperparameter to tune the weightage of the discriminator feature map reconstruction loss. 
- These hyperparameters help to scale and balance the three losses in the generator (reconstruction, divergence, GAN).

- The discriminator's job is easier than the generator's. So we slow down the discriminator's learning rate using a sigmoid based on how much the generator is beating it by.

### Import packages
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# %matplotlib inline
from IPython import display
import pandas as pd
import tensorflow_probability as tfp
ds = tfp.distributions

"""### Set archcitecture parameters"""

TRAIN_BUF=60000
BATCH_SIZE=64
TEST_BUF=10000
DIMS = (28,28,1)
N_Z = 64
N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)

sigmasqr_by_2 = np.float32(3)                                                       # sigma hyper-parameter for the RBF kernel function

"""### Define MAEGAN model class"""

class MAEGAN(tf.keras.Model):

    def __init__(self, **kwargs):                                                   # Constructor for MAEGAN model
        super(VAEGAN, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)                                    # Get Encoder model directly
        self.dec = tf.keras.Sequential(self.dec)                                    # Get Decoder/Generator model directly

        inputs, disc_l, outputs = self.disc_function()                              # The discriminator is created using a function defined later.
        self.disc = tf.keras.Model(inputs=[inputs], outputs=[outputs, disc_l])      # The function returns input, logit_output and l_layer output layers of the discriminator, which we tie together to get a model.

        self.disc_lr = tf.Variable(self.lr_base_disc, trainable=False)

        self.enc_optimizer = tf.keras.optimizers.Adam(learning_rate= self.lr_base_gen, beta_1=0.5)      #Initialize optimizer for encoder with Learning Rate = lr_base_gen
        self.dec_optimizer = tf.keras.optimizers.Adam(learning_rate= self.lr_base_gen, beta_1=0.5)      #Initialize optimizer for decoder with Learning Rate = lr_base_gen
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate= self.disc_lr, beta_1=0.5)         #Initialize optimizer for discriminator with dynamic Learning Rate = get_lr_d()

    def encode(self, x):                                                        # Use Encoder model to encode x into N(0, 1) samples
        qz_samples = self.enc(x)
        return qz_samples

    def get_lr_d(self):
        return ( self.lr_base_disc * self.decay_coeff )                         # To get discriminator's learning rate.

    def update_lr_d(self):
        self.disc_lr.assign(self.get_lr_d())                                    # Function to update and set the disc_lr after each epoch 

    def decode(self, z):                                                        # Decode/Generate sample from a point in latent space
        return self.dec(z)

    def discriminate(self, x):                                                  # Get the discriminator models output on data x.
        return self.disc(x)

    def reconstruct(self, x):                                                   # Reconstruct images (For testing)
        samp = self.encode(x)
        return self.decode(samp)

    def compute_loss(self, x):
        qz_samples = self.encode(x)                                                    # q_z gets the encoded samples from distriobution (encoder output as a distribution) for x
 
        pz_samples = tf.random.normal(shape=(BATCH_SIZE, N_Z), mean=0.0, stddev=1.0)      # We get samples of the prior p_z = N(0,1)

        xg = self.decode(qz_samples)                                                          # xg = reconstructed x
        z_samp = tf.random.normal([x.shape[0], 1, 1, qz_samples.shape[-1]])                   # z is a sample from the prior N(0, 1)
        xg_samp = self.decode(z_samp)                                                # xg_sample is the sample data generated from the sample of prior, i.e., z

        d_x, ld_x = self.discriminate(x)                                        # We discriminate on x (real data). Probability output = d_x. L_layer output = ld_x        
        
        d_xg, ld_xg = self.discriminate(xg)                                     # We discriminate on reconstruction of  x. Probability output = d_xg. L_layer output = ld_xg

        d_xg_samp, ld_xg_samp = self.discriminate(xg_samp)                      # We discriminate on generated (aux) sample. Probability output = d_xg_samp. L_layer output = ld_xg_samp

        # GAN losses
        disc_real_loss = compute_gan_loss(logits=d_x, is_real=True)                     # Discriminator loss on real images. [ disc_real_loss = -log(Dis(x)) ]
        disc_recn_loss = compute_gan_loss(logits=d_xg, is_real=False)                   # Discriminator loss on reconstructed images [ disc_recn_loss = -log(1-Dis(Dec(Enc(x)))) ] (Missing in original work)
        disc_fake_loss = compute_gan_loss(logits=d_xg_samp, is_real=False)              # Discriminator loss on generated images. [ disc_fake_loss = -log(1-Dis(Dec(z)))

        gen_recn_loss = compute_gan_loss(logits=d_xg, is_real=True)                     # Generator loss on reconstructions
        gen_fake_loss = compute_gan_loss(logits=d_xg_samp, is_real=True)                # Generator loss on fake samples

        gen_loss = gen_fake_loss

        discrim_layer_recon_loss = (                                                  # Compute the feature map reconstruction loss on reconstructed images
            tf.reduce_mean(tf.reduce_mean(tf.math.square(ld_x - ld_xg), axis=0))
            / self.beta_
        )

        self.decay_coeff = sigmoid((disc_fake_loss - gen_fake_loss), mult=self.alpha_)            # decay_coeff is the multiplier to discriminator's learning rate
                                                                                                    # A sigmoid based on how much discriminator is beating the generator by
                                                                                                    # To balance generator and discriminator learning

        latent_loss = get_mmd(pz_samples, qz_samples) / self.lambda_                 # MMD_DIVERGENCE LOSS

        return (
            self.decay_coeff,
            latent_loss,
            discrim_layer_recon_loss,
            gen_loss,
            disc_recn_loss,
            disc_fake_loss,
            disc_real_loss,
        )

    # @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
            (
                _,
                latent_loss,
                discrim_layer_recon_loss,
                gen_loss,
                disc_recn_loss,
                disc_fake_loss,
                disc_real_loss,
            ) = self.compute_loss(x)

            enc_loss = latent_loss + discrim_layer_recon_loss
            dec_loss = gen_loss + discrim_layer_recon_loss                      
            disc_loss = disc_fake_loss + disc_real_loss                         # GAN loss

        enc_gradients = enc_tape.gradient(enc_loss, self.enc.trainable_variables)
        dec_gradients = dec_tape.gradient(dec_loss, self.dec.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return enc_gradients, dec_gradients, disc_gradients

    @tf.function
    def apply_gradients(self, enc_gradients, dec_gradients, disc_gradients):
        self.enc_optimizer.apply_gradients(
            zip(enc_gradients, self.enc.trainable_variables)
        )
        self.dec_optimizer.apply_gradients(
            zip(dec_gradients, self.dec.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    def train(self, x):
        enc_gradients, dec_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(enc_gradients, dec_gradients, disc_gradients)


def compute_gan_loss(logits, is_real=True):
    """Computes standard gan loss between logits and labels
                
        Arguments:
            logits {[type]} -- output of discriminator
        
        Keyword Arguments:
            isreal {bool} -- whether labels should be 0 (fake) or 1 (real) (default: {True})
        """
    if is_real:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)

    return tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits
    )


def sigmoid(x, mult=20):
    """ squashes a value with a sigmoid
    """
    return tf.constant(1.0) / (
        tf.constant(1.0) + tf.exp(-(mult*x))
    )


# MMD functions
def rbf_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def get_mmd(x, y, sigma_sqr=1.0):
    x_kernel = rbf_kernel(x, x)
    y_kernel = rbf_kernel(y, y)
    xy_kernel = rbf_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

"""### Define the network architecture"""

encoder = [                                                                               # Define encoder network structure
    tf.keras.layers.InputLayer(input_shape=DIMS),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=N_Z*2),
    tf.keras.layers.Dense(units=N_Z)
]

decoder = [                                                                               # Define decoder network structure
    tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    ),
]

def critic():                                                                             # Define the discriminator function using a CNN classifier 
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            )(inputs)
    conv2 = tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation="relu"
            )(conv1)
    flatten = tf.keras.layers.Flatten()(conv2)
    lastlayer = tf.keras.layers.Dense(units=512, activation="relu")(flatten)
    outputs = tf.keras.layers.Dense(units=1, activation = None)(lastlayer)
    return inputs, lastlayer, outputs

"""### Instantiate MAEGAN Model"""

model = MAEGAN(
    enc = encoder,
    dec = decoder,
    disc_function = critic,
    lr_base_gen = 1e-3,                               # Base learning rate for encoder and decoder networks 
    lr_base_disc = 1e-3,                              # Base learning rate for discriminator network
    lambda_ = 1e-4,                                     
    beta_ = .01,                                      # lambda, beta hyper-parameters as described above   
    alpha_ = 0.5                                      # alpha_ is a hyper-parameter determining the behaviour of the sigmoid function defined above. 
)

"""### Load Dataset"""

(train_images, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

# batch datasets
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).batch(BATCH_SIZE))

"""### Train the model"""

# Sample some images as example data
example_data = next(iter(train_dataset))

# Test trainer
model.train(example_data)

# Creating a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns=[
    'decay_coeff',
    'latent_loss',
    'discrim_layer_recon_loss',
    'gen_loss',
    'disc_recn_loss',
    'disc_fake_loss',
    'disc_real_loss',
])

print(len(losses.columns))
print(losses.columns)

n_epochs = 50

#Train epochs
for epoch in range(n_epochs):

    # train
    for batch, train_x in tqdm(
        zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
    ):
        model.train(train_x)
    # test on holdout
    loss = []
    for batch, test_x in tqdm(
        zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
    ):
        loss.append(model.compute_loss(test_x))
        
    #print(len(np.mean(loss, axis=0)))
    losses.loc[len(losses)] = np.mean(loss, axis=0)

    model.update_lr_d()                                                         # Slow down the learning rate of the discriminator after each epoch
