use std::cmp;

use nn::{layers, objectives, optimizers};
use nn::formatter::Formatter;
use nn::training_results::TrainingResults;
use linalg::{Matrix};

use std::io::prelude::*;
use std::fs::File;
use std::{io, str, fmt};
use std::io::Read;
use std::str::FromStr;
use std::path::Path;
use std::f32;
use std::env;
use std::fs;

pub struct TrainOptions {
    pub shuffle: bool,
    pub epochs: u64,
    pub batch_size: u64
}

impl TrainOptions {
    pub fn default() -> TrainOptions {
        TrainOptions {
            shuffle: true,
            epochs: 1,
            batch_size: 64
        }
    }

    pub fn with_epochs(mut self, epochs: u64) -> TrainOptions {
        self.epochs = epochs;
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> TrainOptions {
        self.shuffle = shuffle;
        self
    }

    pub fn with_batch_size(mut self, batch_size: u64) -> TrainOptions {
        self.batch_size = batch_size;
        self
    }
}

pub struct Network<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone> {
    layers: Vec<Box<layers::Layer>>,
    objective: Obj,
    optimizer: Opt,
    output: Box<Out>,
    formatter: Box<Formatter>
}

impl<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone> Network<Out, Obj, Opt> {
    pub fn new(layers: Vec<Box<layers::Layer>>, objective: Obj,
               optimizer: Opt, output: Box<Out>, formatter: Box<Formatter>) -> Network<Out, Obj, Opt> {
        Network {
            layers: layers,
            objective: objective,
            optimizer: optimizer,
            output: output,
            formatter: formatter
        }
    }

    pub fn layers_count(&self) -> usize {
        self.layers.len()
    }

    pub fn get_layer(&self, index: usize) -> &Box<layers::Layer> {
        &self.layers[index]
    }

    pub fn update_weights(&mut self, index: usize, weights: &Vec<f64>) -> &mut Box<layers::Layer> {
        &mut self.layers[index].update_weights(&weights);
        &mut self.layers[index]
    }

    pub fn get_mut_layer(&mut self, index: usize) -> &mut Box<layers::Layer> {
        &mut self.layers[index]
    }

    pub fn fit(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>, train_options: TrainOptions) {
        for i in 0..train_options.epochs {
            self.formatter.output_epoch_start(i + 1, train_options.epochs);
            self.process_and_run_epoch(input, expected, &train_options);
            self.formatter.output_epoch_end(i + 1, train_options.epochs);
        }
    }

    fn process_and_run_epoch(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>, train_options: &TrainOptions) {
        if train_options.shuffle {
            let mut cloned_input = input.clone();
            let mut cloned_expected = expected.clone();
            self.shuffle(&mut cloned_input, &mut cloned_expected);
            self.run_epoch(&cloned_input, &cloned_expected, train_options);
        } else {
            self.run_epoch(input, expected, train_options)
        }
    }

    fn run_epoch(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>, train_options: &TrainOptions) {
        let rows = input.rows as u64;
        let total_batches = (rows / train_options.batch_size) + ((rows % train_options.batch_size != 0) as u64);
        let mut results = TrainingResults::default();
        results.total_count = input.rows as u64;
        for n in 0..total_batches {
            let start = (n * train_options.batch_size) as usize;
            let end = cmp::min(train_options.batch_size * (n + 1), rows) as usize;
            let x = input.slice_rows(start..end);
            let y = expected.slice_rows(start..end);

            let (hit_count, miss_count, loss) = self.train_on_batch(&x, &y);
            self.update_result(&mut results, (end - start) as u64, hit_count, miss_count, loss);
            self.formatter.output_results(&results);
        }
    }

    fn update_result(&self, results: &mut TrainingResults, count: u64, hit_count: u64, miss_count: u64, loss: f64) {
        results.current_count += count;
        results.hit_count += hit_count;
        results.miss_count += miss_count;
        results.total_loss += loss;
    }

    pub fn shuffle<T: Clone>(&self, input: &mut Matrix<T>, expected: &mut Matrix<T>) {
        let swaps = input.shuffle_rows();
        for (row, other) in swaps {
            expected.swap_rows(row, other);
        }
    }

    pub fn train_on_batch(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>) -> (u64, u64, f64) {
        let results = self.forward(input);
        let gradients = self.backward(&results, expected);
        let ref optimizer = self.optimizer.clone();
        for (index, gradient) in gradients {
            let mut weights = self.get_mut_layer(index).get_mut_weights();
            //rows:3(data.row) col: 128,128,10 layers; elements len:100352(784*128),16384, 1280 ;true
            write_weights(&weights.elements, index);
            let normalized_gradient = gradient.transform(|v| v / (input.rows as f64));
            optimizer.apply_gradients(weights, &normalized_gradient);
        }
        let last = results.last().unwrap();
        let loss = self.loss_from_probs(&last, expected);
        let (hit_count, miss_count) = self.hit_miss_from_probs(&last, expected);
        (hit_count, miss_count, loss)
    }

    pub fn hit_miss_from_probs<T: From<u8> + Clone + PartialEq>(&self, probs: &Matrix<f64>, expected: &Matrix<T>) -> (u64, u64)
            where f64: From<T> {
        let expected_normalized = self.objective.predict_from_probs(&expected.cast());
        self.objective.predict_from_probs(probs).reduce_with_index((0, 0), |(hit, miss), v, row, _col| {
            if { expected_normalized.at(row, 0) == v } { (hit + 1, miss) } else { (hit, miss + 1) }
        })
    }

    pub fn accuracy_from_probs<T: From<u8> + Clone + PartialEq>(&self, probs: &Matrix<f64>, expected: &Matrix<T>) -> f64
            where f64: From<T> {
        let (hit, miss) = self.hit_miss_from_probs(probs, expected);
        (hit as f64) / (hit as f64 + miss as f64)
    }

    pub fn predict(&mut self, input: &Matrix<f64>) -> Matrix<u8> {
        let probs = self.predict_probs(input);
        self.objective.predict_from_probs(&probs)
    }

    pub fn predict_probs(&mut self, input: &Matrix<f64>) -> Matrix<f64> {
        //load weights
        for i in (0..6).step_by(2) {
            let weights: Vec<f64> = load_weights(i);
            // self.get_mut_layer(i as usize).get_mut_weights() = weights;
            // self.layers[i as usize];
            let mut i_usize: usize = i as usize;
            // let t = self.get_mut_layer(i_usize);
            // t.update_weights(&weights);
            self.update_weights(i_usize, &weights);
            // self.get_mut_layer(i_usize).update_weights(&weights);
            // self.layers[i_usize].update_weights(&weights);
            // for layer in self.layers.iter() {
            //     let next = layer.update_weights(&weights);
            // }

        }

        let results = self.forward(input);
        let output = results.last().unwrap();
        output.clone()
    }

    pub fn accuracy<T: From<u8> + Clone + PartialEq>(&mut self, input: &Matrix<f64>, expected: &Matrix<T>) -> f64
            where f64: From<T> {
        let probs = self.predict_probs(input);
        self.accuracy_from_probs(&probs, expected)
    }

    pub fn mean_loss(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        self.loss(input, expected) / (input.rows as f64)
    }

    pub fn loss(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        let probs = self.predict_probs(input);
        self.loss_from_probs(&probs, expected)
    }

    pub fn mean_loss_from_probs(&self, predictions: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        self.loss_from_probs(predictions, expected) / (predictions.rows as f64)
    }

    pub fn loss_from_probs(&self, predictions: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        self.objective.loss(&predictions, expected).reduce(0.0, |acc, v| acc + v)
    }

    pub fn forward(&self, input: &Matrix<f64>) -> Vec<Matrix<f64>> {
        let mut results = vec![input.clone()];
        for layer in self.layers.iter() {
            let next = layer.compute(&results.last().unwrap());
            results.push(next);
        }
        let next = self.output.compute(&results.last().unwrap());
        results.push(next);
        results
    }

    pub fn backward(&self, results: &Vec<Matrix<f64>>, expected: &Matrix<f64>) -> Vec<(usize, Matrix<f64>)> {
        let mut gradients: Vec<(usize, Matrix<f64>)> = vec![];
        let mut back_results = vec![self.objective.delta(&results[results.len() - 1], expected)];
        let last_layer_index = self.layers_count();
        for i in (0..last_layer_index).rev() {
            let gradient = self.layers[i].delta(&results[i + 1], &back_results[back_results.len() - 1]);
            if self.layers[i].has_trainable_weights() {
                let gradient = results[i].t().matmul(&back_results[back_results.len() - 1]);
                gradients.push((i, gradient));
            }
            back_results.push(gradient);
        }
        gradients.reverse();
        gradients
    }
}
pub fn write_weights(w: &Vec<f64>, index: usize) -> std::io::Result<()> {
    let index_str = index.to_string();
    let file_name = format!(
        "weights_{}.txt", index_str
    );
    let mut buffer = File::create(file_name)?;
    let space = " ";
    // let mut buffer_str: String = w.into_iter().map(|&i| i.to_string().push_str(&space)).collect::<String>();
    let mut final_str = String::from("");
    for wi in w.iter() {
        let mut st_w:String = wi.to_string();
        &final_str.push_str(&st_w);
        &final_str.push_str(space);
    }
    final_str.pop();
    let bytes_written = buffer.write(final_str.as_bytes());
    Ok(())
}

pub fn load_weights(index: i32) -> Vec<f64> {
    let mut weights: Vec<f64> = Vec::new();
    let index_str = index.to_string();
    let file_name = format!(
        "weights_{}.txt", index_str
    );
    if let Ok(lines) = read_lines(file_name) {
        for line in lines {
            if let Ok(ip) = line {
                let temp: Vec<&str> = ip.split(' ').collect();
                let mut j  = 0;
                for t in temp {
                    let num_k: f64 = t.to_string().parse().unwrap();
                    // print!("{},{}",index,num_k);
                    &weights.push(num_k);
                }

            }
        }
    }
    weights
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
