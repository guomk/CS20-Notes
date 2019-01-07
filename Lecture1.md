# tf.Session()

* A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.
* A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.
* Session will also allocate memory to store the current values of variables.

``` python
with tf.Session() as sess:
```

or

```python
sess = tf.Session()
...
sess.close()
```

## Distributed Computation

```python
# Creates a graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
  c = tf.multiply(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

## Building more than one graph will cause trouble

* Multiple graphs require multiple sessions, each will try to use all available resources by default
* Can't pass data between them without passing them through python/numpy, which doesn't work in distributed
* It’s better to have disconnected subgraphs within one graph

# tf.Graph() (not recommended to build more than one graph)

* to add operators to a graph, set it as default:

    ```python
    g = tf.Graph()
    with g.as_default():
        x = tf.add(3, 5)
    sess = tf.Session(graph=g)
    with tf.Session() as sess:
    sess.run(x)
    ```
* to handle the default graph:

    ```python
    g = tf.get_default_graph()
    ```

* to handle the default graph and user created graph:

    ```python
    g1 = tf.get_default_graph()
    g2 = tf.Graph()
    # add ops to the default graph
    with g1.as_default():
        a = tf.Constant(3)
    # add ops to the user created graph
    with g2.as_default():
        b = tf.Constant(5)
    ```