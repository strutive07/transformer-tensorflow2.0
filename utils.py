import os
import tensorflow as tf
import time
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Mask:
    @classmethod
    def create_padding_mask(cls, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]
    
    @classmethod
    def create_look_ahead_mask(cls, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    
    @classmethod
    def create_masks(cls, input, target):
        encoder_padding_mask = Mask.create_padding_mask(input)
        decoder_padding_mask = Mask.create_padding_mask(input)
        
        look_ahead_mask = tf.maximum(
            Mask.create_look_ahead_mask(tf.shape(target)[1]),
            Mask.create_padding_mask(target)
            )
        
        return encoder_padding_mask, look_ahead_mask, decoder_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def label_smoothing(target_data, depth, epsilon=0.1):
    target_data_one_hot = tf.one_hot(target_data, depth=depth)
    n = target_data_one_hot.get_shape().as_list()[-1]
    return ((1 - epsilon) * target_data_one_hot) + (epsilon / n)


class Trainer:
    def __init__(
            self,
            model,
            dataset,
            loss_object=None,
            optimizer=None,
            checkpoint_dir='./checkpoints',
            batch_size=None,
            distribute_strategy=None,
            vocab_size=32000,
            epoch=20,
    ):
        self.batch_size = batch_size
        self.distribute_strategy = distribute_strategy
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.vocab_size = vocab_size
        self.epoch = epoch
        self.dataset = dataset

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        #         self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        if self.optimizer is None:
            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)
        else:
            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')

    def multi_gpu_train(self, reset_checkpoint=False):
        with self.distribute_strategy.scope():
            self.dataset = self.distribute_strategy.experimental_distribute_dataset(self.dataset)
            self.trainer(reset_checkpoint=reset_checkpoint, is_distributed=True)

    def single_gpu_train(self, reset_checkpoint=False):
        self.trainer(reset_checkpoint=reset_checkpoint, is_distributed=False)

    def trainer(self, reset_checkpoint, is_distributed=False):
        current_day = datetime.datetime.now().strftime("%Y%m%d")
        train_log_dir = './logs/gradient_tape/' + current_day + '/train'
        os.makedirs(train_log_dir, exist_ok=True)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if not reset_checkpoint:
            if self.checkpoint_manager.latest_checkpoint:
                print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

            self.checkpoint.restore(
                self.checkpoint_manager.latest_checkpoint
            )
        else:
            print("reset and initializing from scratch.")

        for epoch in range(self.epoch):
            start = time.time()
            print('start learning')

            for (batch, (input, target)) in enumerate(self.dataset):
                if is_distributed:
                    loss = self.distributed_train_step(input, target)
                else:
                    loss = self.train_step(input, target)

                self.checkpoint.step.assign_add(1)
                if batch % 50 == 0:
                    print(
                        "Epoch: {}, Batch: {}, Loss:{}, Accuracy: {}".format(epoch, batch, self.train_loss.result(),
                                                                             self.train_accuracy.result()))
                if batch % 10000 == 0 and batch != 0:
                    self.checkpoint_manager.save()
            print("{} | Epoch: {} Loss:{}, Accuracy: {}, time: {} sec".format(
                datetime.datetime.now(), epoch, self.train_loss.result(), self.train_accuracy.result(),
                time.time() - start
            ))
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('train_accuracy', self.train_accuracy.result(), step=epoch)

            self.checkpoint_manager.save()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.validation_loss.reset_states()
            self.validation_accuracy.reset_states()
        self.checkpoint_manager.save()

    def basic_train_step(self, input, target):
        target_include_start = target[:, :-1]
        target_include_end = target[:, 1:]
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            input, target_include_start
        )

        with tf.GradientTape() as tape:
            pred = self.model.call(
                input=input,
                target=target_include_start,
                input_padding_mask=encoder_padding_mask,
                look_ahead_mask=look_ahead_mask,
                target_padding_mask=decoder_padding_mask,
                training=True
            )

            loss = self.loss_function(target_include_end, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(target_include_end, pred)
        if self.distribute_strategy is None:
            return tf.reduce_mean(loss)
        else:
            return loss

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        real_one_hot = label_smoothing(real, depth=self.vocab_size)
        loss = self.loss_object(real_one_hot, pred)

        mask = tf.cast(mask, dtype=loss.dtype)

        loss *= mask
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, input, target):
        return self.basic_train_step(input, target)

    @tf.function
    def distributed_train_step(self, input, target):
        loss = self.distribute_strategy.experimental_run_v2(self.basic_train_step, args=(input, target))
        loss_value = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        return tf.reduce_mean(loss_value)


def translate(input, data_loader, trainer, seq_max_len_target=100):
    if data_loader is None:
        ValueError('data loader is None')

    if trainer is None:
        ValueError('trainer is None')

    if trainer.model is None:
        ValueError('model is None')

    if not isinstance(seq_max_len_target, int):
        ValueError('seq_max_len_target is not int')

    encoded_data = data_loader.encode_data(input, mode='source')
    encoded_data = data_loader.texts_to_sequences([encoded_data])
    encoder_input = tf.convert_to_tensor(
        encoded_data,
        dtype=tf.int32
    )
    decoder_input = [data_loader.dictionary['target']['token2idx']['<s>']]
    decoder_input = tf.expand_dims(decoder_input, 0)
    decoder_end_token = data_loader.dictionary['target']['token2idx']['</s>']

    for i in range(seq_max_len_target):
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            encoder_input, decoder_input
        )
        pred = trainer.model.call(
            input=encoder_input,
            target=decoder_input,
            input_padding_mask=encoder_padding_mask,
            look_ahead_mask=look_ahead_mask,
            target_padding_mask=decoder_padding_mask,
            training=False
        )
        pred = pred[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)

        if predicted_id == decoder_end_token:
            break
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    total_output = tf.squeeze(decoder_input, axis=0)
    return data_loader.sequences_to_texts([total_output.numpy().tolist()], mode='target')