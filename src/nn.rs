use std::cell::RefCell;

use crate::matrix::*;
use crate::utils::*;

const MIN_RAND : f64 = 0.0;
const MAX_RAND : f64 = 0.1;


#[derive(Debug)]
pub struct NeuralNetWork {
	pub layers : Vec<RefCell<Layer>>,
	nb_layer : usize,
	input_size : usize,
	cost_function : fn(&Vec<f64>,&Vec<f64>)->f64,
	cost_derivative : fn(f64,f64) -> f64,
}

impl NeuralNetWork {

	pub fn new(config : &Vec<u32>, cost_str : &str, activation_str : &str, output_activation_str :&str)-> NeuralNetWork {
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

		for elem in &config[1..config.len()-1] {
			nn.add(*elem as usize,activation_str);
		}

		nn.add(*config.last().unwrap() as usize, output_activation_str);

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
			grad_w : Matrix::new(nb_neurons, cols),
			grad_b : Matrix::new(nb_neurons, 1),
		});

		self.layers.push(layer);
		self.nb_layer+=1
	}

	pub fn print_output(&self){
		self.layers.last().unwrap().borrow().post_activation.dump();
	}

	pub fn input(&mut self,input : &Vec<f64>){
		assert!(input.len()==self.input_size,"input should have the same lenght");
		assert!(self.layers.len()!=0,"the network should have at least two layers (input and output)");
		
		self.layers[0].borrow_mut().input_pass(input);

		for i in 1..self.nb_layer {
			self.layers[i].borrow_mut().layer_pass(&self.layers[i-1].borrow_mut().post_activation)
		}

	}

	pub fn train(&mut self,data:&[(Vec<f64>,Vec<f64>)],mini_batch_size : usize,epochs: usize,learning_rate : f64,verbose : bool){

		let mut lr_calculated = learning_rate;
		let mut cost_array :Vec<f64> = vec![];
		for epoch in 0..epochs{

			let data_chunks = data.chunks(mini_batch_size);
			let chunks_size = data_chunks.len();

			let mut i = 0;
			if verbose {
				let (datum_input,datum_output) = &data[0];
				self.input(&datum_input);
				let cost = (self.cost_function)(&self.layers.last().unwrap().borrow().post_activation.values,&datum_output);
				display_progress(i, chunks_size,cost,epoch,epochs);
			}
	
			for data in data_chunks {
	
				for layer in &self.layers {
					layer.borrow_mut().grad_w.zero();
					layer.borrow_mut().grad_b.zero();
				}
				self.update_minibatch(data,lr_calculated);
				let (datum_input,datum_output) = &data[0];
				self.input(&datum_input);
	
	
				if verbose && i%50==0 {
					let cost = (self.cost_function)(&self.layers.last().unwrap().borrow().post_activation.values,&datum_output);
					println!("\x1b[3F");
					display_progress(i, chunks_size,cost,epoch,epochs);
				}
	
				i+=1
			}

			let cost = self.batch_cost(&data[0..mini_batch_size.min(data.len())]);
			cost_array.push(cost);

			if verbose {
				
				println!("\x1b[3F");
				display_progress(i, chunks_size,cost,epoch,epochs);
			}
			
			if cost_array.len()>3 && cost_array.last().unwrap() > &cost_array[cost_array.len()-2] {
					lr_calculated/=2.0;
					println!("Changed learning rate ");
			}
		}
		
	}

	fn update_minibatch(&mut self,data:&[(Vec<f64>,Vec<f64>)],learning_rate : f64){

		//compute the gradient sum overt the mini batch
		for (input,output) in data {
			self.input(input);
			
			//last layer
			let last_layer = self.layers.last().unwrap();
			last_layer.borrow_mut().compute_delta_last_layer(output, self.cost_derivative);
			

			last_layer.borrow_mut().compute_w_grad(&self.layers[self.layers.len()-2].borrow().post_activation.values);
			last_layer.borrow_mut().compute_b_grad();

			//others layers exepct the first one
			for i in (1..self.nb_layer-1).rev() {
				let prev_layer = &self.layers[i-1];
				let follow_layer = &self.layers[i+1];
				let layer = &self.layers[i]; 
				//delta calculation
				layer.borrow_mut().compute_delta(&follow_layer.borrow());
				layer.borrow_mut().compute_w_grad(&prev_layer.borrow().post_activation.values);
				layer.borrow_mut().compute_b_grad();
			};

			//first layer (using input)
			let follow_layer = &self.layers[1];
			let layer = &self.layers[0];

			layer.borrow_mut().compute_delta(&follow_layer.borrow());
			layer.borrow_mut().compute_w_grad(input);
		}



		//aplied the meaned gradient to the network
		let mean_value = data.len() as f64;
		//let mut layer_index = 0;


		// println!("{:_^50}","postactivation");
		// self.layers.last().unwrap().borrow().post_activation.dump();
		// println!("{:_^50}","");
		// println!("");

		for layer in &self.layers{
			// println!("{:=^10}",layer_index);
			// layer.borrow().w_matrix.dump();
			// println!("{:=^10}","");
			layer.borrow_mut().update_parameters(mean_value,learning_rate);
		}

	}

	pub fn batch_cost(&mut self,data : &[(Vec<f64>,Vec<f64>)]) -> f64 {
		let mut cost = 0.0;
		let mean_divider = data.len() as f64;

		for (datum_input,datum_output) in data {

			self.input(&datum_input);
			cost += (self.cost_function)(&self.layers.last().unwrap().borrow().post_activation.values,&datum_output);
		};
		cost /= mean_divider;
		cost
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
	grad_w : Matrix<f64>,
	grad_b : Matrix<f64>,
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

	pub fn compute_delta_last_layer(&mut self,output : &Vec<f64>,cost_derivative :fn(f64,f64) -> f64)
	{
		self.post_activation.cost_derivative_mut(output,cost_derivative);
		self.pre_acvtivation.apply_mut(self.activation_derivative);
		self.post_activation.multiply_by_mut(&self.pre_acvtivation);
	}

	pub fn compute_delta(&mut self,following_layer : &Self){
		following_layer.w_matrix.trans_dot(&mut self.post_activation,&following_layer.post_activation);
		self.pre_acvtivation.apply_mut(self.activation_derivative);
		self.post_activation.multiply_by_mut(&self.pre_acvtivation);
	}

	pub fn compute_w_grad(&mut self,prev_layer_values : &Vec<f64>)
	{
		self.grad_w.matrix_weight_compute(prev_layer_values,&self.post_activation.values);
		
	}

	pub fn compute_b_grad(&mut self)
	{
		self.grad_b.add_mut(&self.post_activation);
	}

	pub fn update_parameters(&mut self, mean_value : f64, learning_rate : f64){
		for (grad,bias) in self.grad_b.values.iter_mut().zip(self.b_matrix.values.iter_mut()){
			*bias -= *grad*learning_rate/mean_value;
		}

		for (grad,weight) in self.grad_w.values.iter_mut().zip(self.w_matrix.values.iter_mut()) {
			*weight -= *grad*learning_rate/mean_value;

		}


		// println!("{:=^10}","BIASES");
		// self.b_matrix.dump();
		// println!("{:=^10}","");

		// println!("{:=^10}","WEIGHTs");
		// self.w_matrix.dump();
		// println!("{:=^10}","");
		// println!("");
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
	1.0 / (1.0 + ((-x).exp()))
}

fn d_sigmoid(x : f64) -> f64
{
	sigmoid(x) * (1.0- sigmoid(x))
}


fn identity(x : f64) -> f64
{
	x
}

fn d_indentity(_:f64) -> f64
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
	//x.iter().zip(y.iter()).map(|(&a, &b)|(b-a).abs()).sum::<f64>() 
	x.iter().zip(y.iter()).map(|(&a, &b)|(b-a).powf(2.0)).sum::<f64>()
}

fn d_quadratic_cost(x:f64, y:f64) -> f64
{
	2.0*(x-y)
}