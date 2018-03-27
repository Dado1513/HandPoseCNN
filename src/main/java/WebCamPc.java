import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamDevice;
import org.datavec.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.io.IOException;

public class WebCamPc {

    public static void main(String[] args) throws IOException {
        Webcam webcam = Webcam.getDefault();
        webcam.setViewSize(new Dimension(320,240));
        webcam.open();

        ImageIO.write(webcam.getImage(), "PNG", new File("hello-world.png"));
        ImageLoader loader = new ImageLoader();
        INDArray image = loader.asMatrix(webcam.getImage());
        webcam.close();
    }
}
