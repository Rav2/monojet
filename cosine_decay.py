# based on https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
import tensorflow as tf
from tensorflow.keras import backend as K

def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=500,
                           start_lr=0.0,
                           target_lr=1e-3,
                           final_lr=1e-5):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    learning_rate = final_lr+ 0.5 * target_lr * (1.0 + tf.cos(tf.constant(np.pi) * tf.convert_to_tensor(global_step - warmup_steps - hold, dtype=np.float32) / tf.convert_to_tensor(total_steps - warmup_steps - hold, dtype=np.float32)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    
    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)

    return learning_rate



class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold, final_lr):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold
        self.final_lr = final_lr
    
    def get_config(self):
        config = {
          'start_lr': self.start_lr,
          'target_lr': self.target_lr,
          'warmup_steps': self.warmup_steps,
          'total_steps': self.total_steps,
          'hold': self.hold,
          'final_lr':self.final_lr,
        }
        return config

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(global_step=tf.cast(step, np.float32),
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold,
                                    final_lr=self.final_lr)

        return tf.where(
            step > self.total_steps, self.final_lr, lr, name="learning_rate"
        )
