import tensorflow as tf

invalid_map = tf.Variable([2.0])
check = tf.Variable([3.0])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run((invalid_map+check)/2))
