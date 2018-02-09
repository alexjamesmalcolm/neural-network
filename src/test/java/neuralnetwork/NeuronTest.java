package neuralnetwork;

import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertThat;

import java.util.Map;
import java.util.TreeMap;

import org.junit.Test;

public class NeuronTest {

	@Test
	public void shouldHaveSumFunctionForInputsAndWeightsOne() {
		Neuron underTest = new Neuron();
		Map<Integer, Double> inputs = new TreeMap<>();
		inputs.put(0, 1.0);
		inputs.put(1, 1.0);
		inputs.put(2, 0.0);
		Map<Integer, Double> weights = new TreeMap<>();
		weights.put(0, 0.2);
		weights.put(1, -0.5);
		weights.put(2, 0.4);
		double sum = underTest.SumFunction(inputs, weights);
		assertThat(sum, is(-0.3));
	}

	@Test
	public void shouldHaveSumFunctionForInputsAndWeightsTwo() {
		Neuron underTest = new Neuron();
		Map<Integer, Double> inputs = new TreeMap<>();
		inputs.put(0, 1.0);
		inputs.put(1, 1.0);
		inputs.put(2, 1.0);
		Map<Integer, Double> weights = new TreeMap<>();
		weights.put(0, 0.2);
		weights.put(1, 0.5);
		weights.put(2, 0.7);
		double sum = underTest.SumFunction(inputs, weights);
		assertThat(sum, is(1.4));
	}
	
	@Test
	public void shouldHaveSigmoidFunctionReduceToLessThanOne() {
		Neuron underTest = new Neuron();
		double num = underTest.sigmoidFunction(0.1);
		assertThat(num, is(allOf(lessThan(1.0), greaterThan(0.5))));
	}
	
	@Test
	public void shouldHaveSigmoidFunctionIncreaseNegativeToGreaterThanNegative1() {
		Neuron underTest = new Neuron();
		double num = underTest.sigmoidFunction(-0.1);
		assertThat(num, is(allOf(lessThan(0.5), greaterThan(0.0))));
	}
}
