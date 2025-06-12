function showHome() {
  document.getElementById("mainContent").innerHTML = `
    <div class="header">
      <h1>EZ-MLAI</h1>
      <h2>Easy Access to Machine Learning AI</h2>
        <h2>No Matter Your Experience Level</h2>
    </div>
    <h3><b>Want to use AI, but don't know how to code? Or are you an experienced coder who doesn't want to be hassled with intricacies?</b></h3>
    <small>Then EZ-MLAI is for you. You can customize machine-learning models for an assortment of the most popular models and get model metrics in graphs and digits, all with just a dataset and no coding</small>
    <h3><h3>
    <img src="assets/images/UserPathwayFinal.png" style="width: 700px; height: 600px;" alt="User Pathway">
    <h4>EZ-MLAI provides various accommodations and allows for easy machine-learning model building. All you need is a dataset that fits the model parameters listed under each specific model. We provide choices from a list of the 6 most popular Machine Learning algorithms to choose from to build your model. Build Your Novel Machine Learning Model Today.</h4>
    <p>Explore our machine-learning models by navigating to the Models page and learn more about key vocab from the Model Descriptions page.</p>
      <h4>Data and the corresponding Models that you should use:</h4>
      <p>Fully numerical features with a continuous numeric target: Linear Regression</p>
      <p>Mixed categorical and numerical features with a numeric target: Random Forest</p>
      <p>Complex relationships or non-linear patterns in tabular data: Neural Network</p>
      <p>A mix of numerical and categorical independent variables and categorical dependent variables: Either K-Nearest-Neighbor or Naive Bayes</p>
      <p>Numerical features with a categorical target: Support Vector Machine</p>
      <div class="bottom-space"></div>
  `;
}

function showModelOptions() {
  const models = [
    { id: 'knn', name: 'K-Nearest Neighbor', img: 'KNN Example.png'},
    { id: 'svm', name: 'Support Vector Machine', img: 'SVM Example.png'},
    { id: 'naive_bayes', name: 'Naive Bayes', img: 'Naïve Bayes Example.png'},
    { id: 'neural_network', name: 'Neural Network', img: 'Neural Network Example.png',},
    { id: 'random_forest', name: 'Random Forest', img: 'Random Forest Example.png'},
    { id: 'linear_regression', name: 'Linear Regression', img: 'Linear Regression Example.png'}
  ];

  const cards = models.map(model => `
    <div class="model-section" style="background-image: url('assets/images/${model.img}')">
      <button class="model-button" onclick="openModelPopup('${model.id}', '${model.name}')">Upload Dataset</button>
    </div>
  `).join('');

  document.getElementById("mainContent").innerHTML = `
    <h2>Select a Model</h2>
    ${cards}
  `;
  
}

function showDescriptions() {
  document.getElementById("mainContent").innerHTML = `
    <h2>Model, Metrics, and Visualization Descriptions</h2>

        <div style="margin-top: 20px; border: 2px solid #000; padding: 10px; border-radius: 5px;">
            <h3>Linear Regression</h3>
            <p>Linear regression is a statistical method that models the relationship between a dependent variable and one or more independent variables.</p>
            <ul>
                <li><b>Metrics:</b></li>
                <ul>
                    <li><b>MSE (Mean Squared Error):</b> Measures the average squared difference between actual and predicted values.</li>
                    <li><b>MAE (Mean Absolute Error):</b> Measures the average absolute difference between actual and predicted values.</li>
                    <li><b>RMSE (Root Mean Squared Error):</b> Measures the square root of the average squared difference between actual and predicted values.</li>
                    <li><b>EVS (Explained Variance Score):</b> Measures the proportion of variance explained by the model.</li>
                    <li><b>R-squared:</b> Measures the proportion of variance in the dependent variable that is predictable from the independent variables.</li>
                    <li><b>Adjusted R-squared:</b> Adjusts the R-squared value for the number of predictors in the model.</li>
                    <li><b>Equation of Model:</b> The mathematical representation of the relationship between dependent and independent variables.</li>
                </ul>
                <li><b>Visualizations:</b></li>
                <ul>
                    <li><b>Predicted vs Actual:</b> Plot the predicted values against the actual values to assess the model's accuracy.</li>
                    <li><b>Residual Plot:</b> Shows the residuals (errors) between the actual and predicted values.</li>
                    <li><b>Q-Q Plot:</b> Compares the distribution of the residuals to a normal distribution.</li>
                    <li><b>Distribution of Data Histograms:</b> Displays the distribution of each variable in the dataset.</li>
                    <li><b>Scatter Plot of Data vs Dependent Variable:</b> Plots each independent variable against the dependent variable to visualize relationships.</li>
                    <li><b>Loss Chart:</b> Shows the loss value over iterations during training.</li>
                </ul>
            </ul>
        </div>

        <div style="margin-top: 20px; border: 2px solid #000; padding: 10px; border-radius: 5px;">
            <h3>Random Forest</h3>
            <p>Random forest is an ensemble learning method for classification, regression, and other tasks, that operates by constructing multiple decision trees.</p>
            <ul>
                <li><b>Metrics:</b></li>
                <ul>
                    <li><b>MSE (Mean Squared Error):</b> Measures the average squared difference between actual and predicted values.</li>
                    <li><b>MAE (Mean Absolute Error):</b> Measures the average absolute difference between actual and predicted values.</li>
                    <li><b>RMSE (Root Mean Squared Error):</b> Measures the square root of the average squared difference between actual and predicted values.</li>
                    <li><b>R-squared:</b> Measures the proportion of variance in the dependent variable that is predictable from the independent variables.</li>
                    <li><b>Adjusted R-squared:</b> Adjusts the R-squared value for the number of predictors in the model.</li>
                    <li><b>MAPE (Mean Absolute Percentage Error):</b> Measures the average absolute percentage difference between actual and predicted values.</li>
                    <li><b>Explained Variance Score:</b> Measures the proportion of variance explained by the model.</li>
                </ul>
                <li><b>Visualizations:</b></li>
                <ul>
                    <li><b>Feature Importance:</b> Shows the importance of each feature in the model.</li>
                    <li><b>Partial Dependence:</b> Plots the marginal effect of a feature on the predicted outcome.</li>
                    <li><b>Residuals:</b> Shows the residuals (errors) between the actual and predicted values.</li>
                    <li><b>Actual vs Predicted:</b> Plots the predicted values against the actual values to assess the model's accuracy.</li>
                    <li><b>Distribution of Numeric Data:</b> Displays the distribution of numeric variables in the dataset.</li>
                </ul>
            </ul>
        </div>

        <div style="margin-top: 20px; border: 2px solid #000; padding: 10px; border-radius: 5px;">
            <h3>Neural Network</h3>
            <p>Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns.</p>
            <ul>
                <li><b>Metrics:</b></li>
                <ul>
                    <li><b>Accuracy:</b> Measures the proportion of correct predictions.</li>
                </ul>
                <li><b>Visualizations:</b></li>
                <ul>
                    <li><b>Confusion Matrix:</b> Shows the performance of the classification model by comparing actual and predicted labels.</li>
                    <li><b>Loss Chart per Epoch:</b> Displays the loss value over epochs during training.</li>
                    <li><b>Training and Validation:</b> Compares the training and validation performance metrics.</li>
                    <li><b>Weights and Biases:</b> Visualizes the learned weights and biases in the network.</li>
                    <li><b>Distribution of Data:</b> Displays the distribution of the data used for training and testing the model.</li>
                </ul>
            </ul>
        </div>

        <div style="margin-top: 20px; border: 2px solid #000; padding: 10px; border-radius: 5px;">
            <h3>K-Nearest Neighbour</h3>
            <p>K-nearest neighbour is a non-parametric method used for classification and regression by comparing the closest training examples in the feature space.</p>
            <ul>
                <li><b>Metrics:</b></li>
                <ul>
                    <li><b>Accuracy:</b> Measures the proportion of correct predictions.</li>
                    <li><b>Precision:</b> Measures the proportion of true positive predictions among all positive predictions.</li>
                    <li><b>Recall:</b> Measures the proportion of true positives identified correctly.</li>
                    <li><b>F1 Score:</b> The harmonic mean of precision and recall.</li>
                    <li><b>AUC-ROC:</b> Measures the ability of the model to distinguish between classes.</li>
                    <li><b>Average Precision:</b> Summarizes the precision-recall curve as the weighted mean of precisions achieved at each threshold.</li>
                </ul>
                <li><b>Visualizations:</b></li>
                <ul>
                    <li><b>Confusion Matrix:</b> Shows the performance of the classification model by comparing actual and predicted labels.</li>
                    <li><b>Predicted vs Actual:</b> Plots the predicted values against the actual values to assess the model's accuracy.</li>
                    <li><b>ROC Curve:</b> Displays the performance of the classification model at all classification thresholds.</li>
                    <li><b>Precision-Recall Curve:</b> Shows the trade-off between precision and recall for different threshold values.</li>
                    <li><b>Data Distribution:</b> Displays the distribution of data used in the model.</li>
                </ul>
            </ul>
        </div>

        <div style="margin-top: 20px; border: 2px solid #000; padding: 10px; border-radius: 5px;">
            <h3>Support Vector Machine</h3>
            <p>Support vector machine is a supervised learning model that analyzes data for classification and regression analysis by finding the hyperplane that best separates the data into classes.</p>
            <ul>
                <li><b>Metrics:</b></li>
                <ul>
                    <li><b>Accuracy:</b> Measures the proportion of correct predictions.</li>
                    <li><b>Precision:</b> Measures the proportion of true positive predictions among all positive predictions.</li>
                    <li><b>Recall:</b> Measures the proportion of true positives identified correctly.</li>
                    <li><b>F1 Score:</b> The harmonic mean of precision and recall.</li>
                </ul>
                <li><b>Visualizations:</b></li>
                <ul>
                    <li><b>Predicted vs Actual:</b> Plots the predicted values against the actual values to assess the model's accuracy.</li>
                    <li><b>Confusion Matrix:</b> Shows the performance of the classification model by comparing actual and predicted labels.</li>
                    <li><b>ROC Curve:</b> Displays the performance of the classification model at all classification thresholds.</li>
                    <li><b>Precision-Recall Curve:</b> Shows the trade-off between precision and recall for different threshold values.</li>
                    <li><b>Data Distribution:</b> Displays the distribution of data used in the model.</li>
                </ul>
            </ul>
        </div>

        <div style="margin-top: 20px; border: 2px solid #000; padding: 10px; border-radius: 5px;">
            <h3>Naive Bayes</h3>
            <p>Naive Bayes is a family of simple probabilistic classifiers based on Bayes' theorem with strong independence assumptions between the features.</p>
            <ul>
                <li><b>Metrics:</b></li>
                <ul>
                    <li><b>Accuracy:</b> Measures the proportion of correct predictions.</li>
                    <li><b>Precision:</b> Measures the proportion of true positive predictions among all positive predictions.</li>
                    <li><b>Recall:</b> Measures the proportion of true positives identified correctly.</li>
                    <li><b>F1 Score:</b> The harmonic mean of precision and recall.</li>
                </ul>
                <li><b>Visualizations:</b></li>
                <ul>
                    <li><b>Confusion Matrix:</b> Shows the performance of the classification model by comparing actual and predicted labels.</li>
                    <li><b>ROC Curve:</b> Displays the performance of the classification model at all classification thresholds.</li>
                    <li><b>Precision-Recall Curve:</b> Shows the trade-off between precision and recall for different threshold values.</li>
                    <li><b>Distribution of Data:</b> Displays the distribution of data used in the model.</li>
                </ul>
            </ul>
            </div>
        </div>
  `;
}

function showAbout() {
  document.getElementById("mainContent").innerHTML = `
    <div class="header">
        <h1>EZ-MLAI</h1>
          <h2>Easy Access to Machine Learning AI</h2>
            <h2>No Matter Your Experience Level</h2>
      </div>
    <h2>The Team</h2>
      <img src="assets/images/HeadShot.jpg" alt="About Us Image" style="width: 200px; height: auto; border: 40px solid transparent;">
    <h3>Lead Developer of EZ-MLAI</h3>
    <h5>My name is Kadhir Ponnambalam and I am an enthusiastic data scientist. I want to help share knowledge on Machine Learning and have data science applicable to as many people as possible. My mission is to provide those with little to no coding experience access to a technology that was previously exclusive to coders. In addition, I want experienced coders to be able to approach Machine Learning without the need for all the hard work. I hope that my application can be used by everybody in hopes of further development in applying this technology.</h5>
    <h5>I also made this tool not only to benefit others but also to learn about machine learning algorithms and web development myself. This was my first full stack project that I wanted to share with everyone.<h5>
  `;
}

function openModelPopup(modelId, modelName) {
  document.getElementById("popupTitle").innerText = `Upload CSV for ${modelName}`;
  document.getElementById("modelPopup").dataset.modelId = modelId;
  document.getElementById("modelPopup").style.display = "block";
}

function closePopup(popupId) {
  document.getElementById(popupId).style.display = "none";
}
