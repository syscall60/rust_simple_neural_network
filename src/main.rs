use rust_simple_nn::matrix::*;


fn add_one(vector :&mut Vec<f64>)->(){
	for elem in vector {
		*elem+=1.0;
	}
}

fn sum_all(vector :&Vec<f64>) -> f64{

	vector.iter().sum()
}

fn main(){
	let mut a = Matrix::new_radom_gen_range(2, 2, 0.0, 10.0);
	a.dump();
	println!("===============");

	a.apply_mut(add_one);
	a.dump();

	dbg!(a.apply(sum_all));
}


