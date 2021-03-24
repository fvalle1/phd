from hopfield4py.hopfield import Hopfield
import tensorflow as tf
import logging
logger = logging.getLogger("hopfield")


class diluted_Hopfield(Hopfield):
    def __init__(self, N: int):
        """
        Hopfield model diluted

        :param N: number of spins
        """
        super().__init__(N)
        logger.info("Creating dilued model..")

    
    def load(self, data):
        logger.info("Loading data...")
        self.N = data.shape[1]
        self.P += data.shape[0]
        all_weights = self.load_all_weights(data)
        self._weights = tf.divide(tf.reduce_sum(all_weights, axis=0), self.N)
    
    def __repr__(self):
        return f"Hopfield model with {self.N} neurons and {self.P} memories loaded (max. {round(self.N*0.138)})"


if __name__ == "__main__":
    data = tf.convert_to_tensor([[1,1,1,1,1,1],[1,-1,-1,1,1,-1]])

    model = dilued_Hopfield(6)
    model.load(data)
    print(model)

    corrupted = tf.convert_to_tensor([1,-1,-1,1,1,1], dtype=tf.double)

    print(model.reconstruct(corrupted))

