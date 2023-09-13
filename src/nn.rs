use std::cell::RefCell;

use crate::matrix::*;

const MIN_RAND : f64 = 0.0;
const MAX_RAND : f64 = 10.0;


#[derive(Debug)]
pub struct NeuralNetWork {
	pub layers : Vec<RefCell<Layer>>,
	nb_layer : usize,
	input_size : usize,
	cost_function : fn(&Vec<f64>,&Vec<f64>)->f64,
	cost_derivative : fn(f64,f64) -> f64,
}

impl NeuralNetWork {

	pub fn new(config : &Vec<u32>, cost_str : &str, activation_str : &str, _output_activation_str :&str)-> NeuralNetWork {
		if config.len()<2 {
			panic!("network should at least have 2 layers (input and output)");
		};

		let (cost_function,cost_derivative) = get_cost_from_string(cost_str);

		let mut nn = NeuralNetWork{
			layers : vec![],
			nb_layer : 0,
			input_size : config[0] as usize,
			cost_function : cost_function,
			cost_derivative : cost_derivative,
		};

		for elem in &config[1..] {
			nn.add(*elem as usize,activation_str);
		}

		nn
	}

	pub fn add(&mut self,nb_neurons : usize,activation_str : &str)
	{

		let (activation_function,activation_derivative) = get_actvation_from_string(activation_str);
		if nb_neurons<=0 {
			panic!("Layer should at least have one neuron");
		};

		let cols = match self.layers.last() {
			Some(layer) => layer.borrow().len as usize,
			None => self.input_size
		};


		let layer = RefCell::new(Layer{
			w_matrix : Matrix::new_radom_gen_range(nb_neurons, cols,MIN_RAND, MAX_RAND),
			b_matrix : Matrix::new_radom_gen_range(nb_neurons, 1, MIN_RAND, MAX_RAND),
			pre_acvtivation : Matrix::new(nb_neurons, 1),
			post_activation : Matrix::new(nb_neurons, 1),
			activation_function,
			activation_derivative,
			len	: nb_neurons,
		});

		self.layers.push(layer);
		self.nb_layer+=1
	}

	pub fn print_output(&self){
		self.layers.last().unwrap().borrow().post_activation.dump();
	}

	pub fn input(&mut self,input : &Vec<f64>){
		assert!(input.len()==self.input_size,"input should have the same len");
		assert!(self.layers.len()!=0,"the network should have at least two layers (input and output)");
		
		self.layers[0].borrow_mut().input_pass(input);

		for i in 1..self.nb_layer {
			self.layers[i].borrow_mut().layer_pass(&self.layers[i-1].borrow_mut().post_activation)
		}

	}
}


#[derive(Debug)]
pub struct Layer {
	w_matrix : Matrix<f64>,
	b_matrix : Matrix<f64>,
	pre_acvtivation : Matrix<f64>,
	post_activation : Matrix<f64>,
	activation_function : fn(f64)->f64,
	activation_derivative : fn(f64)->f64,
	len : usize,
}


impl Layer {
	pub fn input_pass(&mut self,input :&Vec<f64>){
		self.w_matrix.dot_vec(&mut self.pre_acvtivation, input);
		self.pre_acvtivation.add_mut(&self.b_matrix);
		self.pre_acvtivation.apply_to(&mut self.post_activation, self.activation_function)
	}

	pub fn layer_pass(&mut self, input : &Matrix<f64>){
		self.w_matrix.dot(&mut self.pre_acvtivation, input);
		self.pre_acvtivation.add_mut(&self.b_matrix);
		self.pre_acvtivation.apply_to(&mut self.post_activation, self.activation_function)
	}

}




/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */


fn get_actvation_from_string(name : &str) -> (fn(f64) -> f64,fn(f64) -> f64)
{

	match name.trim().to_lowercase().as_str() {
		"sigmoid" | "sigmoïd" => return (sigmoid,d_sigmoid),
		"relu" => return (relu,d_relu),
		"id" | "identity" => return (identity,d_indentity),
		"default" => return (relu,d_relu),
		x  => {
			eprintln!("no function named {x}, using default Relu activation function");
			return (relu,d_relu);
		}

	}
}


fn get_cost_from_string(name : &str) -> (fn(&Vec<f64>,&Vec<f64>) -> f64, fn(f64,f64) -> f64){
	match name.trim().to_lowercase().as_str() {
		"quadratic" => return (quadratic_cost , d_quadratic_cost),
		"default" => return (quadratic_cost , d_quadratic_cost),
		x => {
			eprintln!("no function named {x}, using default quadratic cost function");
			return (quadratic_cost,d_quadratic_cost);
		}
	}
}


/* -------------------------------------------------------------------------- */
/*                        Activation and cost functions                       */
/* -------------------------------------------------------------------------- */


/* -------------------------- Activations function -------------------------- */

fn sigmoid(x : f64) -> f64
{
	1.0/(1.0+x.exp())
}

fn d_sigmoid(x : f64) -> f64
{
	sigmoid(x) * (1.0- sigmoid(x))
}


fn identity(x : f64) -> f64
{
	x
}

fn d_indentity(x:f64) -> f64
{
	1.0	
}


fn relu(x: f64) -> f64
{
	if x>0.0{
		return x
	}
	return x*0.0;
}

fn d_relu(x: f64) -> f64
{
	if x>0.0{
		return 1.0;
	}
	return 0.0;
}

/* ----------------------------- Cost functions ----------------------------- */

fn quadratic_cost(x : &Vec<f64>, y : &Vec<f64>) -> f64
{

	x.iter().zip(y.iter()).map(|(&a, &b)|(b-a).exp2()).sum::<f64>() * 0.5
}

fn d_quadratic_cost(x:f64, y:f64) -> f64
{
	2.0*(y-x)
}