import tensorflow as tf

class Hopfield():
    def __init__(self, N: int):
        self.N = N
        self.neurons = tf.ones((self.N,1), dtype=tf.int8)
        self._weights = tf.zeros((self.N,self.N), dtype=tf.double)
        
    def load(self, data):
        self.N = data.shape[1]
        all_weights = tf.map_fn(lambda point: tf.map_fn(lambda d_i: tf.map_fn(lambda d_j: d_i*d_j,point),point), data)
        self._weights = tf.divide(tf.reduce_sum(all_weights, axis=0), all_weights.shape[0])
            
    def train(self):
        self.neurons =tf.cast(tf.math.sign(tf.matmul(tf.cast(self._weights, tf.double),tf.cast(self.neurons, tf.double))*2-1), tf.int8)
        
    def reconstruct(self, corrupted):
        self.neurons = tf.reshape(corrupted, (self.N,1))
        self.train()
        return tf.reshape(self.neurons, (self.N,))
    
    def __repr__(self):
        return f"Hopfield model with {len(self.neurons)} neurons"
