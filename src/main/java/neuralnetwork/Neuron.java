package neuralnetwork;

import static java.lang.Math.E;
import static java.lang.Math.pow;

import java.util.Map;

public class Neuron {

	public double SumFunction(Map<Integer, Double> inputs, Map<Integer, Double> weights) {
		Double sum = 0.0;
		for (int i = 0; i < inputs.size(); i++) {
			Double input = inputs.get(i);
			Double weight = weights.get(i);
			sum += input * weight;
		}
		return sum;
	}

	public double sigmoidFunction(double sum) {
		return 1 / (1 + pow(E, -sum));
	}

}
