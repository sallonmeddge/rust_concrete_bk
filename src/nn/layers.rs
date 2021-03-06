use linalg::Matrix;
use nn::functions;

pub trait OutputLayer: Layer {}

pub trait Layer {
    fn compute(&self, incoming: &Matrix<f64>) -> Matrix<f64>;
    fn delta(&self, outgoing: &Matrix<f64>, above: &Matrix<f64>) -> Matrix<f64>;
    fn has_trainable_weights(&self) -> bool {
        false
    }
    fn get_mut_weights(&mut self) -> &mut Matrix<f64> {
        panic!("this layer is not trainable")
    }
    fn get_weights(&self) -> &Matrix<f64> {
        panic!("this layer is not trainable")
    }
    fn update_weights(&mut self, incoming: &Vec<f64>) {
        todo!("this layer is not trainable")
    }
}

#[derive(Debug)]
pub struct Relu {
    threshold: f64
}

impl Relu {
    pub fn new() -> Box<Relu> {
        Box::new(Relu { threshold: 0.0 })
    }

    fn compute_in_out(&self, input: &Matrix<f64>, above: &Matrix<f64>) -> Matrix<f64> {
        let op = |v, row, col| if v > self.threshold { above.at(row, col) } else { self.threshold };
        input.transform_with_index(op)
    }
}

impl Layer for Relu {
    fn compute(&self, incoming: &Matrix<f64>) -> Matrix<f64> {
        self.compute_in_out(incoming, incoming)
    }

    fn delta(&self, outgoing: &Matrix<f64>, above: &Matrix<f64>) -> Matrix<f64> {
        outgoing.assert_same_size(above);
        self.compute_in_out(outgoing, above)
    }
}

#[derive(Debug)]
pub struct Dense {
    pub weights: Matrix<f64>,
    pub input_dim: usize,
    pub output_dim: usize
}

impl Dense {
    pub fn new(input_dim: usize, output_dim: usize) -> Box<Dense> {
        let d = 1.0 / (input_dim as f64).sqrt();
        let matrix = Matrix::<f64>::random(input_dim, output_dim, -d, d);
        Box::new(Dense {
            weights: matrix,
            input_dim: input_dim,
            output_dim: output_dim
        })
    }

    pub fn new_with_weights(weights: &Matrix<f64>) -> Box<Dense> {
        Box::new(Dense {
            weights: weights.clone(),
            input_dim: weights.rows.to_owned(),
            output_dim: weights.columns.to_owned()
        })
    }
    fn upd_weights(&mut self, weights: Matrix<f64>) -> &Self {
        self.weights = weights;
        self
    }
}

impl Layer for Dense {
    fn has_trainable_weights(&self) -> bool {
        true
    }

    fn get_weights(&self) -> &Matrix<f64> {
        &self.weights
    }

    fn get_mut_weights(&mut self) -> &mut Matrix<f64> {
        &mut self.weights
    }

    fn compute(&self, incoming: &Matrix<f64>) -> Matrix<f64> {
        incoming.matmul(&self.weights)
    }

    fn delta(&self, _outgoing: &Matrix<f64>, above: &Matrix<f64>) -> Matrix<f64> {
        above.matmul(&self.weights.t())
    }

    fn update_weights(&mut self, incoming: &Vec<f64>) {
        // let rows = 1;
        // let columns = incoming.len();
        let m = Matrix::new_from(self.weights.rows, self.weights.columns, incoming.to_vec(), true);
        self.weights = m;
        // self.upd_weights(m);
    }
}

#[derive(Clone)]
pub struct Softmax;

impl Softmax {
    pub fn new() -> Box<Softmax> {
        Box::new(Softmax {})
    }
}

impl Layer for Softmax {
    fn compute(&self, incoming: &Matrix<f64>) -> Matrix<f64> {
        functions::softmax(incoming)
    }

    fn delta(&self, outgoing: &Matrix<f64>, above: &Matrix<f64>) -> Matrix<f64> {
        let delta = outgoing * above;
        let sums = delta.reduce_rows(0.0, |acc, v| acc + v);
        delta.transform_with_index(|v, row, col| v - outgoing.at(row, col) * sums.at(row, 0))
    }
}

impl OutputLayer for Softmax {}


#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Box<Sigmoid> {
        Box::new(Sigmoid {})
    }
}

impl Layer for Sigmoid {
    fn compute(&self, incoming: &Matrix<f64>) -> Matrix<f64> {
        incoming.transform(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn delta(&self, outgoing: &Matrix<f64>, above: &Matrix<f64>) -> Matrix<f64> {
        outgoing.transform_with_index(|v, row, col| v * (1.0 - v) * above.at(row, col))
    }
}

impl OutputLayer for Sigmoid {}
