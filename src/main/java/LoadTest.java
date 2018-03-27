import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LoadTest {
    protected static final Logger log = LoggerFactory.getLogger(LoadTest.class);
    protected static int index_result = 87;
    protected static int height = 200;
    protected static int width = 200;
    protected static int channels = 1;

    protected static long seed = 42;//42;

    public static void main(String[] args) throws IOException {
        long timestart = System.currentTimeMillis();

        log.info("Load data....");

        String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/modelSerialize/");

        ArrayList<String> allClassLabels = new ArrayList<String>();
        allClassLabels.add("1");
        allClassLabels.add("2");

        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(basePath + "handposeclassification_result"+index_result+".zip");
        DataNormalization scaler = new VGG16ImagePreProcessor();

        // test image not in dataset
        File file = new File(System.getProperty("user.dir"), "src/main/resources/number2.jpg");
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray image = loader.asMatrix(file);
        scaler.transform(image);
        INDArray output = network.output(image);
        System.out.println("Predict for "+ file.getName()+" --> "+allClassLabels.get(output.argMax(1).getInt(0)));


    }
}
