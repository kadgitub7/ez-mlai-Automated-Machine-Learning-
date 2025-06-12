function uploadAndRunModel() {
  const modelId = document.getElementById("modelPopup").dataset.modelId;

  switch (modelId) {
    case 'knn':
      runKNNModel();
      break;
    case 'svm':
      runSVMModel();
      break;
    case 'naive_bayes':
      runNaiveBayesModel();
      break;
    case 'neural_network':
      runNeuralNetworkModel();
      break;
    case 'random_forest':
      runRandomForestModel();
      break;
    case 'linear_regression':
      runLinearRegressionModel();
      break;
    default:
      document.getElementById("errorMsg").textContent = "Unknown model selected.";
  }
}
function submitPrediction() {
  const formElements = document.querySelectorAll('#predictionForm input');
  const inputValues = Array.from(formElements).map(el => el.value);
  const modelId = document.getElementById("reportPopup").dataset.modelId;

  const payload = {
    model: modelId,
    features: inputValues
  };

  fetch("http://localhost:5000/predict_model", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        document.getElementById("predictionResult").innerHTML = `<p style="color:red">${data.error}</p>`;
      } else {
        document.getElementById("predictionResult").innerHTML = `<p><strong>Prediction:</strong> ${data.prediction}</p>`;
      }
    })
    .catch(err => {
      console.error(err);
      document.getElementById("predictionResult").innerHTML = "An error occurred.";
    });
}

