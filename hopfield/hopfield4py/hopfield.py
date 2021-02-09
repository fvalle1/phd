import tensorflow as tf
import logging
logger = logging.getLogger("hopfield")
hdl = logging.StreamHandler()
hdl.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.setLevel("DEBUG")
logger.addHandler(hdl)

class Hopfield():
    def __init__(self, N: int):
        """
        Create the model

        :param N: number of spins
        """
        logger.info("Creating model..")
        self.N = N
        self.P = 0
        self.neurons = tf.ones((self.N,1), dtype=tf.int64)
        self._weights = tf.zeros((self.N,self.N), dtype=tf.double)
        self.nthreads = 12
        
    @tf.function
    def load_all_weights(self, data):
        return tf.map_fn(lambda point: tf.map_fn(lambda d_i: tf.map_fn(lambda d_j: d_i*d_j,point),point, parallel_iterations=self.nthreads), data, parallel_iterations=self.nthreads)
    
    def load(self, data: tf.Tensor) -> None:
        """
        Train number od memories. The number of memories is automatically inferred from data.shape[0]
        
        :param data: tensor of shape (,Nspins) with data to train
        """
        logger.info("Loading data...")
        self.N = data.shape[1]
        self.P += data.shape[0]
        all_weights = self.load_all_weights(data)
        self._weights = tf.divide(tf.reduce_sum(all_weights, axis=0), self.N)

    @tf.function
    def _sign(self, x):
        return tf.constant(1, tf.float64) if tf.greater(x, tf.constant(0, x.dtype)) else tf.constant(-1, tf.float64)

    def train(self)->None:
        """
        set the weights
        """
        logger.debug("Training...")
        #_sign is required to avoid zeros
        self.neurons = tf.cast(tf.map_fn(self._sign, tf.matmul(tf.cast(self._weights, tf.double), tf.cast(self.neurons, tf.double))), tf.int64)
        self.neurons = tf.reshape(self.neurons, (self.N, 1))

    @tf.function
    def reconstruct(self, corrupted: tf.Tensor)->tf.Tensor:
        """
        Reconstruct a memory

        :param corrupted: Corrupted memory 
        """
        logger.debug("Reconstructing...")
        self.neurons = tf.reshape(corrupted, (self.N,1))
        self.train()
        return tf.reshape(self.neurons, (self.N,))
    
    def __repr__(self):
        return f"Hopfield model with {self.N} neurons and {self.P} memories loaded (max. {round(self.N*0.138)})"


if __name__ == "__main__":
    data = tf.convert_to_tensor([[1,1,1,1,1,1],[1,-1,-1,1,1,-1]])

    model = Hopfield(6)
    model.load(data)
    print(model)

    corrupted = tf.convert_to_tensor([1,-1,-1,1,1,1], dtype=tf.double)

    print(model.reconstruct(corrupted))
