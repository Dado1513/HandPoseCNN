
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;

import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FilenameFilter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2YCrCb;


public class MainHandPose {
    protected static final Logger log = LoggerFactory.getLogger(MainHandPose.class);
    protected static int height = 150;
    protected static int width = 150;
    protected static int channels = 1;
    protected static int numExamples = 21420;

    /// / 192 gesture1 // my_number 1044
    // 192 gesture
    // 800 gesture_2
    // 1044 my_number e my_number_2
    // my_number_200_background_black 551
    // 1400 gesture_complete
    // 166 number_polish
    //1218 my_number_200_white_backround

    protected static int numLabels = 5; // 4 gesture
    protected static int batchSize = 40;

    protected static long seed = 42;//42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 40;
    protected static double splitTrainTest = 0.7;
    protected static int nCores = 2;
    protected static boolean save = true;
    protected static double learningrate = 0.005;

    protected static String modelType = "LeNetAddLayer"; // LeNet, AlexNet or Custom but you need to fill it out

    public void run(String[] args) throws Exception {
        long timestart = System.currentTimeMillis();

        log.info("Load data....");


        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.dir"), "src/main/resources/all_edge_image_150/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);
        // try random path filther
        //RandomPathFilter pathFilter = new RandomPathFilter(rng);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);

        //resize *2
        ImageTransform resizeTransform1 = new ResizeImageTransform(38,38);
        ImageTransform resizeTransform2 = new ResizeImageTransform(50,50);

        //scale*6
        ImageTransform scaleTransform1 = new ScaleImageTransform(3);
        ImageTransform scaleTransform6 = new ScaleImageTransform(5);
        ImageTransform scaleTransform2 = new ScaleImageTransform(2);
        ImageTransform scaleTransform3 = new ScaleImageTransform(1);
        ImageTransform scaleTransform4 = new ScaleImageTransform(1);
        ImageTransform scaleTransform5 = new ScaleImageTransform(4);

        ImageTransform colorTransform1 = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        //ImageTransform resizeTransform = new ResizeImageTransform(80,80);
        //ImageTransform rotationTransform = new RotateImageTransform(45);
        //ImageTransform cropTransform=new CropImageTransform(50);

        List<ImageTransform> transforms = Arrays.asList(flipTransform1,
                warpTransform, scaleTransform4
                ,flipTransform2,scaleTransform6,scaleTransform1,resizeTransform2);
    /*
                /*,scaleTransform5,resizeTransform1,scaleTransform2,scaleTransform3});/*,
                scaleTransform5,scaleTransform3,resizeTransform2,scaleTransform4,resizeTransform1,scaleTransform2,scaleTransform6, rotationTransform});*/

        /*
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         **/
        //DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        DataNormalization scaler = new VGG16ImagePreProcessor();

        log.info("Build model....");

        // Uncomment below to try AlexNet. Note change height and width to at least 100
        // MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

        ConvolutionalNeuralNet cnnFactory = new ConvolutionalNeuralNet(height,width,channels,numLabels,seed,rng,iterations,learningrate);
        MultiLayerNetwork network;

        switch (modelType) {

            case "LeNet":
                network = cnnFactory.lenetModel();
                break;

            case "Custom":
                network = cnnFactory.customModelGitHub();
                break;

            case "MyModel":
                network = cnnFactory.myModelNeural();
                break;

            case "DeepModel":
                network = cnnFactory.myDeepModel();
                break;

            case "PaperNet":
                learningrate = 0.003;
                network = cnnFactory.netPaper();
                break;

            case "VGG19":
                width = 224;
                height = 224;
                network = cnnFactory.getVGG19();
                break;

            case "LeNetAddLayer":
                network = cnnFactory.lenetModelWithAddLayer();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");

        }

        network.init();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        network.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(listenerFreq));
        uiServer.attach(statsStorage);

        // network.setListeners(new ScoreIterationListener(listenerFreq));

        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;




        log.info("Train model....");
        // Train without transformations
        recordReader.initialize(trainData, null);

        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);

        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
        network.fit(trainIter);
/*
        // Train with transformations
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
            network.fit(trainIter);
        }
*/
        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));
        String dirName = System.getProperty("user.dir")+"/src/main/resources/ResultNetVarious/";
        File dirResultLeNet = new File(dirName);

        // / lamba function
        FilenameFilter textFilter = (dir, name) -> { return name.endsWith(".txt"); };
        int index_result = dirResultLeNet.listFiles(textFilter).length;

        long timeend = System.currentTimeMillis();
        long executionTime = timeend -timestart;
        double secondTime = (double)executionTime*Math.pow(10,-3);
        double minutTime = secondTime/60;
        index_result = index_result+1;
        PrintWriter pw = new PrintWriter(dirName+"result_"+index_result+".txt");
        pw.println(network.conf().toJson());
        //pw.println("Solo  " +transforms.size() +" trasformazioni, gesturecompelte 7 classi");
        pw.println("TypeInput: " + pathFilter.getClass().getName());
        pw.println("IMage Preprocessing: "+scaler.getClass().getName());
        pw.println("MainPath: "+mainPath.toString());
        pw.println("Size width: "+width+ ", height: "+height+", channel: "+channels);
        pw.println("Con regolarizzazione true");
        pw.println("Batchsize : " +batchSize);
        pw.println("Epoche : "+epochs);
        pw.println("Split train: " + splitTrainTest*100 +" %");
        pw.println("Execution time :" + minutTime+ " m");
        pw.println("TYpeNet : "+modelType);
        pw.println(eval.stats());
        pw.println();
        pw.println(eval.confusionToString());

        pw.close();

        // Example on how to get predict results with trained model
        dataIter.reset();
        DataSet testDataSet = dataIter.next();

        List<String> allClassLabels = recordReader.getLabels();
        INDArray predictedClasses = network.output(testDataSet.getFeatures());
        System.out.println("True "+allClassLabels.get(testDataSet.getLabels().argMax(1).getInt(0)));
        System.out.println("Predict "+allClassLabels.get(predictedClasses.argMax(1).getInt(0)));

        if (save) {
            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/modelSerialize/");
            ModelSerializer.writeModel(network, basePath + "handposeclassification_result"+index_result+".zip", true);
        }
        log.info("****************Example finished********************");

        // test image not in dataset
        File file = new File(System.getProperty("user.dir"), "src/main/resources/number2_150.jpg");
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray image = loader.asMatrix(file);
        scaler.transform(image);
        INDArray output = network.output(image);
        System.out.println("Predict for "+ file.getName()+" --> "+allClassLabels.get(output.argMax(1).getInt(0)));

        // test image not in dataset
        file = new File(System.getProperty("user.dir"), "src/main/resources/number01.jpg");
        loader = new NativeImageLoader(height, width, channels);
        image = loader.asMatrix(file);
        scaler.transform(image);
        output = network.output(image);
        System.out.println("Predict for "+ file.getName()+" --> "+allClassLabels.get(output.argMax(1).getInt(0)));

        // test image not in dataset
        file = new File(System.getProperty("user.dir"), "src/main/resources/number03.jpg");
        loader = new NativeImageLoader(height, width, channels);
        image = loader.asMatrix(file);
        scaler.transform(image);
        output = network.output(image);
        System.out.println("Predict for "+ file.getName()+" --> "+allClassLabels.get(output.argMax(1).getInt(0)));

        System.exit(0);
    }


    public static void main(String[] args) throws Exception {
        new MainHandPose().run(args);
    }

}
