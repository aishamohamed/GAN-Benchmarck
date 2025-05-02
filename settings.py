IMG_SIZE = 128
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.0002
N_CRITIC = 1
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

def load_data(folder):
    images = []
    for f in os.listdir(folder):
        if f.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, f))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = (img.astype(np.float32) - 127.5) / 127.5
            images.append(img)
    return tf.data.Dataset.from_tensor_slices(np.array(images)).shuffle(1000).batch(BATCH_SIZE)

dataset = load_data(resized_folder)

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 256, input_dim=latent_dim),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

class GANTrainer:
    def __init__(self, model_type='dcgan'):
        self.model_type = model_type
        self.generator = build_generator(LATENT_DIM)
        self.discriminator = build_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5, beta_2=0.999)
        self.d_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5, beta_2=0.999)
        self.g_loss_metric = tf.keras.metrics.Mean()
        self.d_loss_metric = tf.keras.metrics.Mean()

    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            if self.model_type == 'wgan':
                gp = self.gradient_penalty(images, fake_images)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + 10.0 * gp
                g_loss = -tf.reduce_mean(fake_output)
            else:  # dcgan, cgan, bagan
                d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)) + \
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output))

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
            return d_loss, g_loss

    def gradient_penalty(self, real_images, fake_images):
        alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for batch in dataset:
                for _ in range(N_CRITIC if self.model_type == 'wgan' else 1):
                    d_loss, g_loss = self.train_step(batch)
                self.d_loss_metric(d_loss)
                self.g_loss_metric(g_loss)
            print(f"{self.model_type.upper()} - Epoch {epoch+1}, D Loss: {self.d_loss_metric.result():.4f}, G Loss: {self.g_loss_metric.result():.4f}")
            self.d_loss_metric.reset_states()
            self.g_loss_metric.reset_states()
            if (epoch + 1) % 50 == 0:
                self.generate_and_save(epoch + 1)

    def generate_and_save(self, epoch):
        noise = tf.random.normal([16, LATENT_DIM])
        fake_images = self.generator(noise, training=False)
        fake_images = (fake_images * 127.5 + 127.5).numpy().astype(np.uint8)
        for i, img in enumerate(fake_images):
            cv2.imwrite(f'{self.model_type}_generated_epoch{epoch}_{i}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))