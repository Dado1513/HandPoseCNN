import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.model.GoogLeNet;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class ConvolutionalNeuralNet {
    protected static final Logger log = LoggerFactory.getLogger(ConvolutionalNeuralNet.class);
    protected int height = 32;
    protected int width = 32;
    protected int channels = 3;
    protected  int numLabels = 4;

    protected  long seed = 42;//42;
    protected  Random rng = new Random(seed);
    protected  int iterations = 1;

    protected  double learningrate = 0.01;

    public ConvolutionalNeuralNet(int height, int width, int channels, int numLabels, long seed, Random rng, int iterations, double learningrate) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.rng = rng;
        this.iterations = iterations;
        this.learningrate = learningrate;
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {

        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    public MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.01) // di solito è false
                .activation(Activation.RELU)
                .learningRate(learningrate) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP).momentum(0.9)
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork lenetModelWithAddLayer(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.01) // di solito è false
                .activation(Activation.RELU)
                .learningRate(learningrate) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP).momentum(0.9)
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                // change stride from 5 to 3
                .layer(2, conv5x5("cnn2", 100, new int[]{3, 3}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                // layer add from original LeNet
                .layer(4,conv5x5("cnn3",200,new int[]{3,3},new int[]{1,1},0)) // 6x6x200
                .layer(5,maxPool("maxpool3",new int[]{2,2})) // 3x3x200

                .layer(6, new DenseLayer.Builder().nOut(500).build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);


    }
    public MultiLayerNetwork myDeepModel(){

        double nonZeroBias = 0.1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER_LEGACY)
                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningrate)
                .regularization(true)
                .l2(0.01)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, convInit("cnn1", channels, 100, new int[]{5, 5}, new int[]{1, 1}, new int[]{4,4}, nonZeroBias)) // 36x36x100
                .layer(1, maxPool("maxpool1", new int[]{2,2})) // 18x18x100
                .layer(2,conv5x5("cnn2", 200,new int[]{1,1},new int[]{0,0},nonZeroBias)) // 14x14x200
                .layer(3,conv5x5("cnn3",500,new int[]{1,1},new int []{0,0},nonZeroBias)) // 10x10x500
                .layer(4, maxPool("maxpool2", new int[]{2,2})) //5x5x500
                .layer(5, new DenseLayer.Builder().nOut(1024).build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork getVGG19(){


        VGG19 vgg19 = new VGG19(numLabels,seed,iterations);

        MultiLayerNetwork network = new MultiLayerNetwork(vgg19.conf());

        return network;


    }

    public MultiLayerNetwork myModelNeural(){
        double nonZeroBias = 0.1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER_LEGACY)

                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningrate)
                .regularization(true)
                .l2(0.01)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, convInit("cnn1", channels, 200, new int[]{5, 5}, new int[]{1, 1}, new int[]{0,0}, nonZeroBias))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2,conv5x5("cnn2", 500,new int[]{5,5},new int[]{1,1},0))
                .layer(3, maxPool("maxpool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(1024).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork customModelGitHub(){
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningrate)
                .biasLearningRate(2*1e-3)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{4, 4}, nonZeroBias))
                //.layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2,conv5x5("cnn2", 100,new int[]{5,5},new int[]{1,1},0))
                //.layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(3, maxPool("maxpool2", new int[]{2,2}))
                .layer(4,conv3x3("cnn3", 50, nonZeroBias))
                .layer(5, fullyConnected("ffn1", 332, nonZeroBias, dropOut, new GaussianDistribution(0, 0.001)))
                //.layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork netPaper(){

        double biasCnn = 0.2;
        double biasDense = 0.1;

        learningrate = 0.003;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.04))
                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningrate)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .list()
                .layer(0, convInit("cnn1", channels, 16, new int[]{5, 5}, new int[]{1, 1}, new int[]{4, 4}, biasCnn)) //36x36x16
                .layer(1, maxPool("maxpool1",new int[]{2,2})) //18x18x16
                .layer(2, conv5x5("cnn2",32,new int[]{1,1},new int[]{0,0},0.2)) // 14x14x32
                .layer(3, maxPool("maxpool2",new int[]{2,2})) //7x7x32
                .layer(4,new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{1,1}, new int[]{0,0}).name("cnn2").nOut(48).biasInit(biasCnn).build()) //4x4x48
                .layer(5, maxPool("maxpool2",new int[]{2,2})) //2x2x48

                .layer(6, new DenseLayer.Builder().name ("ffn1") .weightInit( WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.2)).nOut(512).biasInit(biasDense).build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }


}
