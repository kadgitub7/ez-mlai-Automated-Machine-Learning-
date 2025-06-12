function renderModelResults(data, modelId) {
  const container = document.getElementById("reportContent");
  container.innerHTML = "";

  document.getElementById("reportPopup").dataset.modelId = modelId;

  // Metrics
  const metricsHTML = Object.entries(data.metrics || {})
    .map(([key, value]) => `<p><strong>${key.replace(/_/g, ' ')}:</strong> ${value?.toFixed(4)}</p>`)
    .join("\n");

  container.innerHTML += `
    <h2>Model Metrics</h2>
    ${metricsHTML}
  `;
  container.innerHTML += `
  <h2>Make a Prediction</h2>
  <div id="predictionForm"></div>
  <button onclick="submitPrediction()">Predict</button>
  <div id="predictionResult"></div>
`;


  // Visualization helper
  function addImage(title, base64) {
    if (base64) {
      container.innerHTML += `<h3>${title}</h3><img src="data:image/png;base64,${base64}" alt="${title}" />`;
    }
  }

  addImage("Confusion Matrix", data.confusion_matrix);
  addImage("Predicted vs Actual", data.predicted_vs_actual);
  addImage("ROC Curve", data.roc_curve);
  addImage("Precision-Recall Curve", data.precision_recall_curve);
  addImage("Target Variable Distribution", data.target_distribution);
  addImage("PCA Visualization", data.pca_plot);
  addImage("Residual Plot", data.residual_plot);

  // Feature Histograms
  if (data.histograms) {
    container.innerHTML += `<h2>Feature Histograms</h2>`;
    Object.entries(data.histograms).forEach(([col, base64]) => {
      addImage(`Histogram of ${col}`, base64);
    });
  }

  // Python Code
  if (data.python_code) {
    container.innerHTML += `
      <h2>Python Code</h2>
      <pre>${escapeHtml(data.python_code)}</pre>
    `;
  }

  if (data.feature_names) {
  const predictionForm = document.getElementById("predictionForm");
  data.feature_names.forEach((feature, i) => {
    predictionForm.innerHTML += `
      <label>${feature}</label>
      <input type="text" id="input-${i}" placeholder="Enter ${feature}">
    `;
  });
}

}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.innerText = text;
  return div.innerHTML;
}

function showLoadingPopup() {
  document.getElementById("loadingPopup").style.display = "block";
}

function hideLoadingPopup() {
  document.getElementById("loadingPopup").style.display = "none";
}
