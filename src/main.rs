use rust_simple_nn::nn::*;
use rand::{self, Rng, seq::SliceRandom, thread_rng};

fn generate_data()->Vec<(Vec<f64>,Vec<f64>)>{
	let mut result = vec![];
	let mut rng = rand::thread_rng();
	for _ in 0..1000000 {

		let input = vec![rng.gen_range(1.0..2.0),rng.gen_range(1.0..2.0)];
		let ouput = function1(&input);
		result.push((input,ouput));
	};

	result
}

fn generate_id()->Vec<(Vec<f64>,Vec<f64>)>{
	let mut result = vec![];
	let mut rng = rand::thread_rng();
	for _ in 0..1000000 {

		let input = vec![rng.gen_range(-1.0..1.0),rng.gen_range(-1.0..1.0),rng.gen_range(-1.0..1.0),rng.gen_range(-1.0..1.0),rng.gen_range(-1.0..1.0)];
		let ouput = function2(&input);
		result.push((input,ouput));
	};

	result
}


fn gen_quadrant()->Vec<(Vec<f64>,Vec<f64>)>{
	let mut result = vec![];
	let mut rng = rand::thread_rng();
	for _ in 0..1000000 {

		let input = vec![rng.gen_range(-10.0..10.0),rng.gen_range(-10.0..10.0)];
		let ouput = some_quadrant(&input);
		result.push((input,ouput));
	};

	result
}

fn generate_xor_data()->Vec<(Vec<f64>,Vec<f64>)>{
	let mut result = vec![];
	for _ in 0..10000 {

		result.push((vec![0.0,0.0],vec![0.0]));
		result.push((vec![0.0,1.0],vec![1.0]));
		result.push((vec![1.0,0.0],vec![1.0]));
		result.push((vec![1.0,1.0],vec![0.0]));
	}

	result
}

//2 inputs, 1 output
fn function1(input : &[f64]) -> Vec<f64>{
	vec![((input[0]+input[1]).powi(2))]
}

//5 inputs, 5 outputs
fn function2(input : &[f64]) -> Vec<f64>{
	vec![
		(input[0]+input[1]),
		(input[0]+input[2]),
		(input[3]+input[1]),
		(input[4]+input[0]-input[3]*4.0),
		(input[4]-input[0]-input[3]),
	]
}

// if the coordoante is in a certain qudrant, then it's 1, otherwise it's 0
fn some_quadrant(input : &[f64])-> Vec<f64>{
	let (x,y) = (input[0],input[1]);
	if x<0.0 && y>0.0 {
		return vec![1.0];
	}

	return vec![0.0];
}

fn main(){


	let mut neural_network = NeuralNetWork::new(
		&vec![2,4,1],
		"default",
		"relu",
		"sigmoid"
	);

	println!("===========================");
	neural_network.input(&vec![1.0,2.0]);
	neural_network.print_output();

	println!("Generating data ....");
	let mut data = gen_quadrant();
	println!("data generated, strating shuffling");
	let mut rng = thread_rng();
	data.shuffle(&mut rng);
	println!("data shuffled, strating training");	
	neural_network.train(&data, 120,100, 0.5,true);
	
	println!("===========================");
	neural_network.input(&vec![10.0,20.0]);
	neural_network.print_output();

	println!("===========================");
	neural_network.input(&vec![-30000.0,2.1]);
	neural_network.print_output();

	println!("===========================");
	neural_network.input(&vec![-2.1,3000000.0]);
	neural_network.print_output();

	println!("===========================");
	neural_network.input(&vec![0.0,0.0]);
	neural_network.print_output();

	println!("===========================");
	neural_network.input(&vec![-10.0,10.0]);
	neural_network.print_output();


}


