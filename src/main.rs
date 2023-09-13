use rust_simple_nn::nn::*;



fn main(){


	let mut neural_network = NeuralNetWork::new(
		&vec![3,3,8,4],
		"default",
		"default",
		"default"
	);

	neural_network.input(&vec![1.0,2.0,3.0]);
	neural_network.print_output();
	println!("===========================");
	neural_network.input(&vec![1.0,2.0,3.01]);
	neural_network.print_output();


}


