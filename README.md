# rust-binary-MNIST

This is a sample network for [binary-nn](https://github.com/ttakamura/binary-nn-rust) crate.

[binary-nn](https://github.com/ttakamura/binary-nn-rust) is a rust crate to build binalized neural-network.


## Learn

You should learn your network by your favorite tool, e.g, TensorFlow, Chainer. Then, export the weights as files.

[Here](https://github.com/ttakamura/binary-net-sandbox/blob/master/chainer-binary-net/export.py) is a sample code to export it.


## Predict

```sh
cargo build --release
./target/release/serialize data/binary_net data/mnist.bin
```
