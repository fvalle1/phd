import tensorflow as tf
from hopfield4py import Hopfield

@tf.function
def hamming(A: tf.Tensor, B: tf.Tensor)-> tf.Tensor:
    """
    Hamming distance sum(abs((A-B)))/lenght(A)

    A and be must have the same shape

    :param A: first tensor
    :param B: second tensor
    """
    assert(A.shape==B.shape)
    return tf.divide(tf.cast(tf.reduce_sum(tf.abs(tf.subtract(tf.cast(A, dtype=tf.int64),tf.cast(B, dtype=tf.int64)))), tf.float64), tf.constant(2., tf.float64)*tf.cast(A.shape[0], tf.float64))
    
@tf.function
def process_sample(sample):
    return tf.cast(sample*tf.constant(2, tf.int8)-tf.constant(1, tf.int8), tf.int8, name="processed_sample")
    
@tf.function
def get_hamming_symmetric(reconstructed, data):
    return tf.stack([hamming(reconstructed, data), hamming(-reconstructed, data)], axis=0, name="hammings")
    
@tf.function
def get_hamming_minimum(reconstructed, data_tensor):
    return tf.reduce_min(tf.map_fn(lambda data: get_hamming_symmetric(reconstructed, data), data_tensor, parallel_iterations=12, fn_output_signature=tf.float64), axis=1)
    
@tf.function
def predict(sample, data_tensor, model):
        reconstructed = model.reconstruct(sample)
        return tf.argmin(get_hamming_minimum(reconstructed, data_tensor))

def get_real_label(df, sample):
    real = df.loc[sample,"tissue"]
    if type(real)!=str:
        real=real.values[0]
    return real

@tf.function
def get_prediction(sample: tf.Tensor, data_tensor: tf.Tensor, model: Hopfield)->tf.Tensor:
    """
    Get the nearest memory

    :param sample: sample to reconstruct
    :param data_tensor: tensor with memories
    :param model: model used to infer
    :return: argmin of the element of data_tensor nearest to sample
    """
    return predict(process_sample(sample), data_tensor, model)

@tf.function
def get_all_prediction(samples: tf.Tensor, data_tensor: tf.Tensor, model: Hopfield)-> tf.Tensor:
    """
    Get the nearest memory

    :param samples: samples to reconstruct
    :param data_tensor: tensor with memories
    :param model: model used to infer
    :return: tensor with list of argmin of the element of data_tensor nearest to each sample
    """
    return tf.map_fn(lambda sample: get_prediction(sample, data_tensor, model), samples, fn_output_signature=tf.int64, parallel_iterations=12)


def get_predicted_labels(classes: list, samples: tf.Tensor, data_tensor: tf.Tensor, model:Hopfield)->list:
    """
    Get the classes predicted for each sample

    :param classes: list of classes names with shape (nclasses,)
    :param samples: samples to reconstruct with shape (nsamples, nspins)
    :param data_tensor: tensor with memories (nclasses, nspins)
    :return: tensor with list of classes
    """
    return list(map(lambda label_idx: classes[label_idx], get_all_prediction(samples, data_tensor, model).numpy()))
