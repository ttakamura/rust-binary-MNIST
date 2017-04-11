extern crate binary_nn;
extern crate binary_nn_mnist;

use std::env;
use binary_nn_mnist::network::MnistNetwork;
use binary_nn::network::Network;
use binary_nn::loader;

fn main() {
  let args: Vec<String> = env::args().collect();
  let serialize_path = args[1].clone();
  let data_path = args[2].clone();

  let network = MnistNetwork::deserialize_from(&serialize_path);
  println!("success deserialize network from {}", &serialize_path);

  let data = loader::load_f32(&data_path);
  let y = network.predict_f32(&data);
  println!("predict as {}", y);
}
