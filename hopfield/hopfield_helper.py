import tensorflow as tf

@tf.function
def hamming(A, B):
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
def get_prediction(sample, data_tensor, model):
    return predict(process_sample(sample), data_tensor, model)

@tf.function
def get_all_prediction(samples, data_tensor, model):
    return tf.map_fn(lambda sample: get_prediction(sample, data_tensor, model), samples, fn_output_signature=tf.int64, parallel_iterations=12)

def get_predicted_labels(classes, samples, data_tensor, model):
    return list(map(lambda label_idx: classes[label_idx], get_all_prediction(samples, data_tensor, model).numpy()))