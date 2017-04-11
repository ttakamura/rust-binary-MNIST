extern crate binary_nn;
extern crate binary_nn_mnist;

use std::env;
use binary_nn_mnist::network::MnistNetwork;
use binary_nn::network::Network;

fn main() {
  let args: Vec<String> = env::args().collect();
  let weights_path = args[1].clone();
  let serialize_path = args[2].clone();

  let network = MnistNetwork::load_plain_weights(&weights_path, 1000);
  println!("success load weights from {}", &weights_path);

  network.serialize_into(&serialize_path);
  println!("success serialize into {}", &serialize_path);
}
