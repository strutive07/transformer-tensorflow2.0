import tensorflow as tf

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
