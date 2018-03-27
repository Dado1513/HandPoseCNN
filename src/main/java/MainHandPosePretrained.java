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
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
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
import org.deeplearning4j.zoo.model.GoogLeNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MainHandPosePretrained {
    protected static final Logger log = LoggerFactory.getLogger(MainHandPosePretrained.class);
    protected static long seed = 42;//42;
    protected static Random rng = new Random(seed);
    protected static double splitTrainTest = 0.7;
    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 3;
    protected static int numExamples = 800; // 192 gesture1
    protected static int numLabels = 3;
    protected static int batchSize = 12;
    protected static boolean save = false;
    protected static int iterations = 1;
    protected static int epochs = 8;
    protected static String modelType = "VGG16";
    protected static int nCores = 2;
    protected double learningrate = 0.01;



    public void run() throws IOException {
        long timestart = System.currentTimeMillis();

        log.info("Load data....");

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.dir"), "src/main/resources/my_number_200/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);

        //RandomPathFilter pathFilter = new RandomPathFilter(rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];
        ZooModel zooModel = new VGG16();
        zooModel.setInputShape(new int[][]{{channels,width,height},{0,0,0}});
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

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

        //ImageTransform colorTransform1 = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        //ImageTransform resizeTransform = new ResizeImageTransform(80,80);
        //ImageTransform rotationTransform = new RotateImageTransform(45);
        //ImageTransform cropTransform=new CropImageTransform(50);

        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1,
                warpTransform, scaleTransform4,
                flipTransform2,scaleTransform6,scaleTransform1,resizeTransform2,
                scaleTransform5,resizeTransform1,scaleTransform2,scaleTransform3});/*,
              scaleTransform5,scaleTransform3,resizeTransform2,scaleTransform4,resizeTransform1,scaleTransform2,scaleTransform6, rotationTransform});*/


        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(learningrate)
                .regularization(true).l2(0.001)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .weightInit(WeightInit.XAVIER_LEGACY)
                .seed(seed)
                .build();

        // to googlenet
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numLabels)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();

        // Attach listener --> see dashboard
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        vgg16Transfer.setListeners(new StatsListener(statsStorage,1),new ScoreIterationListener(iterations));
        uiServer.attach(statsStorage);

        // read test data
        ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIterTest;
        recordReaderTest.initialize(testData);

        dataIterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numLabels);
        dataIterTest.setPreProcessor(TrainedModels.VGG16.getPreProcessor());

        Evaluation eval;
        dataIterTest.reset();
        eval = vgg16Transfer.evaluate(dataIterTest);
        System.out.println("Eval stats BEFORE fit.....");
        System.out.println(eval.stats());
        dataIterTest.reset();

        System.gc();
        //
        ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIterTrain;
        recordReaderTrain.initialize(trainData);
        dataIterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numLabels);
        dataIterTrain.setPreProcessor(TrainedModels.VGG16.getPreProcessor());

        MultipleEpochsIterator trainIter;
        trainIter = new MultipleEpochsIterator(epochs, dataIterTrain);
        vgg16Transfer.fit(trainIter);
/*
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReaderTrain.initialize(trainData, transform);
            dataIterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numLabels);
            dataIterTrain.setPreProcessor(new VGG16ImagePreProcessor());
            trainIter = new MultipleEpochsIterator(epochs, dataIterTrain);
            vgg16Transfer.fit(trainIter);
            System.gc();
        }

*/
        System.out.println("Finish... Evaluation...");
        eval = vgg16Transfer.evaluate(dataIterTest);
        System.out.println(eval.stats()+"\n");

        System.gc();
        // precision 100%
        if (save) {
            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            ModelSerializer.writeModel(vgg16Transfer, basePath + "HandPoseVGG16.zip", true);
        }


        // single test
        dataIterTest.reset();
        DataSet testDataSet = dataIterTest.next();
        List<String> allClassLabels = recordReaderTest.getLabels();
        INDArray predictedClasses = vgg16Transfer.outputSingle(testDataSet.getFeatures());
        System.out.println("True "+allClassLabels.get(testDataSet.getLabels().argMax(1).getInt(0)));
        System.out.println("Predict "+allClassLabels.get(predictedClasses.argMax(1).getInt(0)));

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


        pw.println("Original Net: "+pretrainedNet.conf().toJson());
        pw.println("vgg16Transfer: "+vgg16Transfer.conf().toJson());
        //pw.println("Solo  " +transforms.size() +" trasformazioni, gesturecompelte 7 classi");
        pw.println("Type: VGG16 pretrained with IMageNet");
        pw.println("Learning rate: "+learningrate);
        pw.println("TypeInput: " + pathFilter.getClass().getName());
        pw.println("MainPath: "+mainPath.toString());
        pw.println("Size width: "+width+ ", height: "+height+", channel: "+channels);
        pw.println("Con regolarizzazione false");
        pw.println("Batchsize : " +batchSize);
        pw.println("Epoche : "+epochs);
        pw.println("Split train: " + splitTrainTest*100 +" %");
        pw.println("Execution time :" + minutTime+ " m");
        pw.println("TYpeNet : "+modelType);
        pw.println(eval.stats());
        pw.println();
        pw.println(eval.confusionToString());
        pw.close();


        // test image not in dataset
        File file = new File(System.getProperty("user.dir"), "src/main/resources/number2.jpg");
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray image = loader.asMatrix(file);
        VGG16ImagePreProcessor scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        INDArray output = vgg16Transfer.outputSingle(image);
        System.out.println("Predict for "+ file.getName()+" --> "+allClassLabels.get(output.argMax(1).getInt(0)));

        // test image not in dataset
        file = new File(System.getProperty("user.dir"), "src/main/resources/number01.jpg");
        loader = new NativeImageLoader(height, width, channels);
        image = loader.asMatrix(file);
        scaler.transform(image);
        output = vgg16Transfer.outputSingle(image);
        System.out.println("Predict for "+ file.getName()+" --> "+allClassLabels.get(output.argMax(1).getInt(0)));

        // test image not in dataset
        file = new File(System.getProperty("user.dir"), "src/main/resources/number03.jpg");
        loader = new NativeImageLoader(height, width, channels);
        image = loader.asMatrix(file);
        scaler.transform(image);
        output = vgg16Transfer.outputSingle(image);
        System.out.println("Predict for "+ file.getName()+" --> "+allClassLabels.get(output.argMax(1).getInt(0)));

        System.exit(0);




    }

    public static void main(String[] args) {
        try {
            new MainHandPosePretrained().run();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
