import tensorflow as tf
from tensorflow import keras

# ----------------------------------------------
#                   LAYERS
# ----------------------------------------------
class EncoderLayer(keras.Layer):
    def __init__(self):
        super().__init__
        self.norm1 = keras.layers.LayerNormalization()
        self.mha = keras.layers.MultiHeadAttention()
        self.norm2 = keras.layers.LayerNormalization()
        self.mlp = keras.Sequential([
            keras.layers.Dense(),
            keras.layers.Dense()
        ])
    
    def call(self, x):
        x_ = self.norm1(x)
        x_ = self.mha(x_, x_, x_)
        x = x + x_
        x_ = self.norm2(x)
        x_ = self.mlp(x_)
        return x + x_

# ----------------------------------------------
#                   MODELS
# ----------------------------------------------
class ViT(keras.Model):
    def __init__(self, L: int, emb_dim: int):
        super().__init__
        self.linear_projection = keras.layers.Dense()
        self.encoder = keras.Sequential([EncoderLayer() for _ in range(L)])
        self.mlp = keras.layers.Dense()
        self.positional_emb = keras.layers.Embedding()
    
    def call(self, x):
        # we'd just do something like tf.arange(len(patches))
        # above through the call of self.positional_emb
        # add the flattened linear patches to the positional we get so then it will learn
        x = self.linear_projection(x)
        pos = tf.range(0, len(), 1.)
        pos_emb = self.positional_emb(pos)
        x += pos_emb
        x = self.encoder(x)
        x = self.mlp(x)
        return x
    
    def patchify(self):
        pass


# ----------------------------------------------
#                  FUNCTIONS
# ----------------------------------------------