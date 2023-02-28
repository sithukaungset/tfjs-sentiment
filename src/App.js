import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const start = window.performance.now();
const App = () => {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      const modelUrl = 'https://cdn.jsdelivr.net/gh/sithukaungset/tfjs-model/model.json';
      const model = await tf.loadLayersModel(modelUrl);

      //const model = await tf.loadLayersModel('https://github.com/sithukaungset/tfjs-model/blob/main/model.json');
      setModel(model);
      console.log("model loaded")
    };
    loadModel();
  }, []);

  // const predict = async (event) => {
  //   const image = event.target.files[0];
  //   const imgTensor = await tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).toFloat();
  //   const offset = tf.scalar(127.5);
  //   const normalized = imgTensor.sub(offset).div(offset);
  //   const batched = normalized.expandDims(0);
  //   const prediction = await model.predict(batched).array();
  //   setPredictions(prediction);
  // };
  const predict = async (event) => {
    const image = event.target.files[0];
    const img = new Image();
    const reader = new FileReader();
    reader.onload = (e) => {
      img.src = e.target.result;
      img.onload = async () => {
        const imgTensor = tf.browser.fromPixels(img).toFloat().expandDims();
        const resized = tf.image.resizeBilinear(imgTensor, [256, 256]);
        //const img = await tf.browser.fromPixels(fname);
        
        const offset = tf.scalar(127.5);
        const normalized = resized.sub(offset).div(offset);
        const prediction = await model.predict(normalized).array();
        setPredictions(prediction);
      };
    };
    reader.readAsDataURL(image);
  };
const end = window.performance.now();
console.log("myFunction: " + (end - start) + "ms");

  return (
    <div>
      <h1>Image Classifier</h1>
      <input type="file" accept="image/*" onChange={predict} />
      {predictions && (
        <div>
          {predictions.map((pred, i) => (
            <p key={i}>{pred}</p>
          ))}
        </div>
      )}
    </div>
  );
};

export default App;




