import numpy as np
import tensorflow as tf


class Transformer(tf.keras.Model):
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 encoder_count,
                 decoder_count,
                 attention_head_count,
                 d_model,
                 d_point_wise_ff,
                 dropout_prob):
        super(Transformer, self).__init__()

        # model hyper parameter variables
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model,
        self.d_point_wise_ff = d_point_wise_ff,
        self.dropout_prob = dropout_prob

        self.encoder_embedding_layer = Embedding_layer(input_vocab_size, d_model)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)
        self.decoder_embedding_layer = Embedding_layer(target_vocab_size, d_model)
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)

        self.encoder_layers = [
            EncoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob
            ) for _ in range(encoder_count)
        ]

        self.decoder_layers = [
            DecoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob
            ) for _ in range(decoder_count)
        ]

        self.linear = tf.keras.layers.Dense(target_vocab_size)
            
    def call(self,
             input,
             target,
             input_padding_mask,
             look_ahead_mask,
             target_padding_mask):
        encoder_tensor = self.encoder_embedding_layer(input)
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor)
        
        for i in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[i](encoder_tensor, input_padding_mask)
        
        target = self.decoder_embedding_layer(target)
        decoder_tensor = self.decoder_embedding_dropout(target)
        
        for i in range(self.decoder_count):
            decoder_tensor, _, _ = self.decoder_layers[i](
                decoder_tensor,
                encoder_tensor,
                look_ahead_mask,
                target_padding_mask
            )
        return self.linear(decoder_tensor)
        

        
class EncoderLayer(tf.keras.layers.Layer):
      def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
            super(EncoderLayer, self).__init__()

            # model hyper parameter variables
            self.attention_head_count = attention_head_count
            self.d_model = d_model
            self.d_point_wise_ff = d_point_wise_ff
            self.dropout_prob = dropout_prob

            self.multi_head_attention = MultiHeadAttention(attention_head_count, d_model, dropout_prob)
            self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
            self.layer_nomr_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
                d_attention_dense, 
                d_model,
                dropout_prob
            )
            self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
            self.layer_nomr_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    
      def call(self, input, mask):
            output, attention = self.multi_head_attention(input, input, input, mask)
            output = self.dropout_1(output)
            output = self.layer_nomr_1(tf.add(input, output)) # residual network
            
            output = self.position_wise_feed_forward_layer(output)
            output = self.dropout_2(output)
            output = self.layer_norm_2(tf.add(input, output)) #residual network
            
            return output, attention

        
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(DecoderLayer, self).__init__()
        
        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        
        self.masked_multi_head_attention = MultiHeadAttention(attention_head_count, d_model, dropout_prob)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_nomr_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.encoder_decoder_attention = MultiHeadAttention(attention_head_count, d_model, dropout_prob)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_nomr_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_attention_dense, 
            d_model,
            dropout_prob
        )
        self.dropout_3 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_nomr_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, decoder_input, encoder_output, look_ahead_mask, padding_mask):
        output, attention_1 = self.masked_multi_head_attention(
            decoder_input,
            decoder_input,
            decoder_input,
            look_ahead_mask
        )
        output = self.dropout_1(output)
        query = self.layer_norm_1(tf.add(decoder_input, output)) # residual network
        
        output, attention_2 = self.encoder_decoder_attention(
            query,
            encoder_output,
            encoder_output,
            padding_mask
        )
        output = self.dropout_2(output)
        encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))
        
        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output)
        output = self.layer_norm_3(tf.add(encoder_decoder_attention_output, output)) #residual network
        
        return output, attention_1, attention_2        
            
        
class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model, dropout_prob):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
    
    def call(self, input):
        input = self.w_1(input)
        input = tf.nn.relu(input)
        input = self.dropout(input)
        return self.w_2(input)
        
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(attention_head_count, d_model, dropout_prob):
        super(MultiHeadAttention, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.dropout_prob

        if d_model % attention_head_count != 0:
            raise ValueError(
                f"d_model({d_model}) % attention_head_count({attention_head_count}) is not zero."
                f"d_model must be multiple of attention_head_count."
            )
        
        self.d_h = d_model // attention_head_count
        
        self.w_query = tf.keras.layers.Dense(d_model)
        self.w_key = tf.keras.layers.Dense(d_model)
        self.w_value = tf.keras.layers.Dense(d_model)
        
        self.scaled_dot_product = ScaledDotProductAttention()
        
        self.ff = tf.keras.layers.Dense(d_model)
    
    def call(self, query, key, value, mask=None):
        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)
        
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)
        
        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output)
        
        return self.ff(output), attention
        
    
    def split_head(self, tensor):
        # input tensor: (batch_size, seq_len, d_model)
        return tf.transpose(
            tf.reshape(
                tensor, 
                (tf.shape(tensor)[0], -1, self.attention_head_count, self.d_h)
                # tensor: (batch_size, seq_len_splited, attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
            # tensor: (batch_size, attention_head_count, seq_len_splited, d_h)
        )
    
    def concat_head(self, tensor):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]), 
            (tf.shape(tensor)[0], -1, self.attention_head_count * self.d_h)
        )

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(dropout_prob):
        super(ScaledDotProductAttention, self).__init__()
        
    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(tf.shape(self.query)[-1], dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        
        if mask is not None:
            scaled_attention_score += (mask * -1e9)
        
        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)
        
        return tf.matmul(attention_weight, v), attention_weight

    
class Embedding_layer(tf.keras.layers.Layer):
    def __init__(vocab_size, d_model):
        # model hyper parameter variables
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    
    def call(self, sequences):
        max_sequence_len = tf.shape(sequences)[0]
        
        output = self.embedding(sequences) * tf.sqrt(tf.cast(d_model, dtype=tf.float32))
        output += self.positional_encoding(max_sequence_len)
    
    def positional_encoding(self, max_len):
        pos = np.expand_dims(np.arange(0, max_len), axis=1)
        index = np.expand_dims(np.arange(0, self.d_model), axis=0)
        
        pe = self.angle(pos, index)
        
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])        
        
        pe = np.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float32)
        
    def angle(self, pos, index):
        return pos / np.power(10000, (index - index % 2) / np.float32(self.d_model))
