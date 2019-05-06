# Set up the environment on Windows 10

Following these steps to set up TF environment in Windows 10:

- Step 1. Open Anaconda Prompt.
- Step 2. Run: `conda create -n tf python=3.6`
- Step 3. Run: `activate tf`
- Step 4. Run: `pip install tensorflow=1.13.1`
- Step 5. Create `test.py` with the following content:

    ```python
    import tensorflow as tf

    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    result = a + b

    with tf.Session() as sess:
        print(sess.run(result))
    ```

- Step 6. Run: `python test.py`. The result printed on screen is:

    ```
    [3. 5.]
    ```

On Windows, the program may prompt a warning message:

> 2019-05-06 14:26:59.413082: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

This message can be safely ignored, according to [this post](https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u).