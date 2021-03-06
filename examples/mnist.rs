extern crate simple_nn;

use simple_nn::{nn, utils};
extern crate concrete;
use concrete::*;

fn main() {
    let mut network = nn::NetworkBuilder::new()
        .add(nn::layers::Dense::new(784, 128))
        .add(nn::layers::Relu::new())
        .add(nn::layers::Dense::new(128, 128))
        .add(nn::layers::Relu::new())
        .add(nn::layers::Dense::new(128, 10))
        .add_output(nn::layers::Softmax::new())
        .minimize(nn::objectives::CrossEntropy::new())
        .with(nn::optimizers::SGD::new(0.5))
        .build();

    println!("loading training data...");
    // let x_train = utils::loader::matrix_from_txt("data/t1.txt").unwrap().transform(|v: f64| v / 255.0);
    // let y_train = utils::loader::matrix_from_txt("data/t2.txt").unwrap().to_one_hot(10);
    // let x_train = utils::loader::matrix_from_txt("data/mnist_sample.txt").unwrap().transform(|v: f64| v / 255.0);
    // let y_train = utils::loader::matrix_from_txt("data/mnist_sample_labels.txt").unwrap().to_one_hot(10);
    // let x_train = utils::loader::matrix_from_txt("data/train_x_60000x784_float32.txt").unwrap().transform(|v: f64| v / 255.0);
    // let y_train = utils::loader::matrix_from_txt("data/train_y_60000_int32.txt").unwrap().to_one_hot(10);

    // let train_options = nn::TrainOptions::default().with_epochs(3).with_batch_size(64);
    // network.fit(&x_train, &y_train, train_options);

    // println!("loading test data...");
    // let x_test = utils::loader::matrix_from_txt("data/t1.txt").unwrap().transform(|v: f64| v / 255.0);
    // let y_test = utils::loader::matrix_from_txt("data/t2.txt").unwrap().to_one_hot(10);
    // original:
    // let x_test = utils::loader::matrix_from_txt("data/test_x_10000x784_float32.txt").unwrap().transform(|v: f64| v / 255.0);
    // let y_test = utils::loader::matrix_from_txt("data/test_y_10000_int32.txt").unwrap().to_one_hot(10);
    // // let x_test = utils::loader::matrix_from_txt("data/mnist_sample.txt").unwrap().transform(|v: f64| v / 255.0);
    // // let y_test = utils::loader::matrix_from_txt("data/mnist_sample_labels.txt").unwrap().to_one_hot(10);

    // let predict_probs = network.predict_probs(&x_test);
    // let loss = network.mean_loss_from_probs(&predict_probs, &y_test);
    // let accuracy = network.accuracy_from_probs(&predict_probs, &y_test);
    // println!("accuracy = {}, mean loss = {}", accuracy, loss);

    //concrete
    let secret_key = LWESecretKey::new(&LWE128_630);
    let encoder = Encoder::new(0., 10., 8, 1).unwrap();
    // LWE::encode_encrypt(&secret_key, v / 255.0, &encoder).unwrap()
    let x2_test = utils::loader::matrix_from_txt("data/t1.txt")
        .unwrap().transform_with_concrete(|v: f64|  v / 255.0, secret_key, encoder);
    // let y2_test = utils::loader::matrix_from_txt("data/t2.txt").unwrap().to_one_hot(10);

    // let predict_probs = network.predict_probs(&x2_test);
    // let loss = network.mean_loss_from_probs(&predict_probs, &y2_test);
    // let accuracy = network.accuracy_from_probs(&predict_probs, &y2_test);
    // println!("accuracy = {}, mean loss = {}", accuracy, loss);
}
