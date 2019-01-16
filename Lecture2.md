# First Tensorflow Program

## Suppress warning

```python
 import os
 os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
```

## Visualization with TensorBoard

```bash
$ python3 [yourprogram].py
$ tensorboard --logdir="./graphs" --port 6006 6006 or any port you want
Then open your browser and go to: http://localhost:6006/
Run it
```

**A demo program [tfboard.py](../code/tfboard.py)**

# Fast evaluation

```python
tf.InteractiveSession()
```

then
`python_var.eval()`

# Constants

> tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

## Broadcasting

```python
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='mul')

# >> [[0 2]
#     [4 6]]
```

## Tensors filled with a specific value

```python
tf.zeros(shape, dtype=tf.float32, name=None)

tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)

tf.ones(shape, dtype=tf.float32, name=None)

tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

tf.fill(dims, value, name=None)
 tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
# In numpy, this is done by 1. create a np array a 2. a.fill()
```

## Constants as sequences

```python
tf.lin_space(start, stop, num, name=None)
tf.lin_space(10.0, 13.0, 4) ==> [10. 11. 12. 13.]

tf.range(start, limit=None, delta=1, dtype=None, name='range')
 tf.range(3, 18, 3) ==> [3 6 9 12 15]
 tf.range(5) ==> [0 1 2 3 4]
```

## Randomly Generated Constants

```python
tf.set_random_seed(seed)
tf.random_normal
tf.truncated_normal
tf.random_uniform
tf.random_shuffle
tf.random_crop
tf.multinomial
tf.random_gamma
```

### Graph Def

`print(sess.graph.as_graph_def())`

# Variables

* Problem with constants: Makes loading graphs expensive when constants are big

## Use `tf.get_variable()`

```python
tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None
)

# Some examples
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
```

## `tf.Variable()` is not recommended

```python
x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...) # and more 
```

## Variable Initialization

* Initializer is an op. You need to execute it within the context of a session

```python
# Initializing all variables at once
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# Initialize a subset of variables
with tf.Session() as sess:
    sess.run(tf.variables_initializer([a, b]))

# Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
  
```

## `eval()`

`print(W.eval()) == print(sess.run(W))

## `tf.Variable.assign()`

### operations needs to be executed in a session to take effect

```python
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())    # >> 10

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print(W.eval())    # >> 100
```

Similar operations are `assign_add()` and `assign_sub()` etc

### Each session maintains its own copy of variables

```python
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))  # >> 20
print(sess2.run(W.assign_sub(2)))   # >> 8

print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50)))  # >> -42

sess1.close()
sess2.close()
```

## Control Dependencies

`tf.Graph.control_dependencies(control_inputs)`

```python
# your graph g have 5 ops: a, b, c, d, e
g = tf.get_default_graph()

with g.control_dependencies([a, b, c]):
    # 'd' and 'e' will only run after 'a', 'b', and 'c' have executed. 
    d = ...
    e = ...
```

# Placeholders

`tf.placeholder(dtype, shape=None, name=None)`

## Feed multiple data points

* Do it one at a time

```python
with tf.Session() as sess:
    for a_value in list_of_values_for_a:
        print(sess.run(c, {a: a_value}))
```

## Feedable tensor

* You can feed_dict any feedable tensor. Placeholder is just a way to indicate that something must be fed
* `tf.Graph.is_feedable(tensor)` True if and only if tensor is feedable

### Feeding values to TF ops

* Helpful for testing. Feed in dummy values to test parts o fa large graph

```python
# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)
with tf.Session() as sess:
    # compute the value of b given a is 15
    sess.run(b, feed_dict={a: 15}) # >> 45
```

# Lazy Loading

* Normal Loading

```python
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) # create the node before executing the graph

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)
writer.close()
```

* Lazy Loading

```python
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code (not a smart act)
writer.close()
```

## Solution

1. Separate definition o fops from computing/running ops
2. Use Python property to ensure function is also loaded once the first time it is called