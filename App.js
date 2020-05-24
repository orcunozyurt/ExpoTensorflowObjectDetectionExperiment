import React, { useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  StatusBar,
  Image,
  TouchableOpacity,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import { fetch } from "@tensorflow/tfjs-react-native";
import * as mobilenet from "@tensorflow-models/mobilenet";
import Constants from "expo-constants";
import * as Permissions from "expo-permissions";
import * as ImagePicker from "expo-image-picker";
import * as jpeg from "jpeg-js";
import { Camera } from "expo-camera";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { ExpoWebGLRenderingContext } from "expo-gl";

const TensorCamera = cameraWithTensors(Camera);
const AUTORENDER = false;

let textureDims;
if (Platform.OS === "ios") {
  textureDims = {
    height: 1920,
    width: 1080,
  };
} else {
  textureDims = {
    height: 1200,
    width: 1600,
  };
}

class App extends React.Component {
  state = {
    isTfReady: false,
    isModelReady: false,
    hasPermission: null,
    predictions: null,
    image: null,
  };

  async componentDidMount() {
    await tf.ready();
    this.setState({ isTfReady: true });
    console.log(this.state.isTfReady);

    this.model = await mobilenet.load();
    this.setState({ isModelReady: true });

    this.getPermissionAsync();
  }

  getPermissionAsync = async () => {
    const { status } = await Camera.requestPermissionsAsync();
    this.setState({ hasPermission: status === "granted" });
    if (status !== "granted") {
      alert("Sorry, we need camera permissions to make this work!");
    }
  };

  imageToTensor(rawImageData) {
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];

      offset += 4;
    }

    return tf.tensor3d(buffer, [height, width, 3]);
  }

  classifyImage = async () => {
    try {
      const imageAssetPath = Image.resolveAssetSource(this.state.image);
      const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
      const rawImageData = await response.arrayBuffer();
      const imageTensor = this.imageToTensor(rawImageData);
      const predictions = await this.model.classify(imageTensor);
      this.setState({ predictions });
      console.log(predictions);
    } catch (error) {
      console.log(error);
    }
  };

  selectImage = async () => {
    try {
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3],
      });

      if (!response.cancelled) {
        const source = { uri: response.uri };
        this.setState({ image: source });
        this.classifyImage();
      }
    } catch (error) {
      console.log(error);
    }
  };

  renderPrediction = (prediction) => {
    return (
      <Text key={prediction.className} style={styles.text}>
        {prediction.className}
      </Text>
    );
  };

  handleImageTensorReady = async (images, updatePreview, gl) => {
    const loop = async () => {
      if (!AUTORENDER) {
        updatePreview();
      }
      const imageTensor = images.next().value;
      const flipHorizontal = Platform.OS === "ios" ? false : true;
      const predictions = await this.model.classify(imageTensor);

      this.setState({ predictions });

      if (!AUTORENDER) {
        gl.endFrameEXP();
      }

      requestAnimationFrame(loop);
    };
    loop();
  };

  render() {
    const { isTfReady, isModelReady, predictions, image } = this.state;
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" />
        <View style={styles.loadingContainer}>
          <Text style={styles.commonTextStyles}>
            TFJS ready? {isTfReady ? <Text>✅</Text> : ""}
          </Text>

          <View style={styles.loadingModelContainer}>
            <Text style={styles.text}>Model ready? </Text>
            {isModelReady ? (
              <Text style={styles.text}>✅</Text>
            ) : (
              <ActivityIndicator size="small" />
            )}
          </View>
        </View>
        {isModelReady ? (
          <TensorCamera
            // Standard Camera props
            style={[styles.camera]}
            type={Camera.Constants.Type.back}
            zoom={0}
            // tensor related props
            cameraTextureHeight={textureDims.height}
            cameraTextureWidth={textureDims.width}
            resizeHeight={200}
            resizeWidth={152}
            resizeDepth={3}
            onReady={this.handleImageTensorReady}
            autorender={AUTORENDER}
          />
        ) : (
          <ActivityIndicator size="small" />
        )}
        <View style={styles.modelResults}>
          {isModelReady && image && (
            <Text style={styles.text}>
              Predictions: {predictions ? "" : "Predicting..."}
            </Text>
          )}
          {isModelReady &&
            predictions &&
            predictions.map((p) => this.renderPrediction(p))}
        </View>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  cameraContainer: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    width: "100%",
    height: "100%",
    backgroundColor: "#fff",
  },
  camera: {
    position: "absolute",
    left: 50,
    top: 100,
    width: 600 / 2,
    height: 800 / 2,
    zIndex: 1,
    borderWidth: 1,
    borderColor: "black",
    borderRadius: 0,
  },
  predictionWrapper: {
    height: 100,
    width: "100%",
    flexDirection: "column",
  },
  modelResults: {
    position: "absolute",
    left: 50,
    top: 100,
    width: 600 / 2,
    height: 800 / 2,
    zIndex: 20,
    borderWidth: 1,
    borderColor: "black",
    borderRadius: 0,
  },
  recordingButton: {
    position: "absolute",
    left: 50,
    bottom: 150,
  },
});

export default App;
