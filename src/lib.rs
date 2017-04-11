#[macro_use]
extern crate serde_derive;
extern crate binary_nn;

pub mod network {
  // use binary_nn::backend::bitmatrix_trait::*;
  // use binary_nn::layer::base::Layer;
  use binary_nn::layer::linear::BinaryLinearLayer;
  use binary_nn::layer::batch_norm::BatchNormLayer;
  use binary_nn::network::Network;

  #[derive(Serialize, Deserialize, Debug, PartialEq)]
  pub struct MnistNetwork {
    l1: BinaryLinearLayer,
    bn1: BatchNormLayer,
    l2: BinaryLinearLayer,
    bn2: BatchNormLayer,
    l3: BinaryLinearLayer,
    bn3: BatchNormLayer,
  }

  impl Network for MnistNetwork {}

  impl MnistNetwork {
    // load plain weight from given path.
    //
    // following code is export the files:
    // https://github.com/ttakamura/binary-net-sandbox/blob/master/chainer-binary-net/export.py
    //
    pub fn load_plain_weights(prefix: &str, hidden_unit: u32) -> Self {
      let l1 = BinaryLinearLayer::load((prefix.to_string() + ".l1.W.dat").as_str(),
                                       hidden_unit,
                                       784);
      let l2 = BinaryLinearLayer::load((prefix.to_string() + ".l2.W.dat").as_str(),
                                       hidden_unit,
                                       hidden_unit);
      let l3 = BinaryLinearLayer::load((prefix.to_string() + ".l3.W.dat").as_str(), 10, hidden_unit);

      let bn1 = BatchNormLayer::load((prefix.to_string() + ".b1.dat").as_str(),
                                     hidden_unit as usize);
      let bn2 = BatchNormLayer::load((prefix.to_string() + ".b2.dat").as_str(),
                                     hidden_unit as usize);
      let bn3 = BatchNormLayer::load((prefix.to_string() + ".b3.dat").as_str(), 10);

      return MnistNetwork {
        l1: l1,
        bn1: bn1,
        l2: l2,
        bn2: bn2,
        l3: l3,
        bn3: bn3,
      };
    }
  }
}
