function runSVMModel() {
  const fileInput = document.getElementById("modelFileInput");
  const targetColumnInput = document.getElementById("targetColumnInput");
  const errorMsg = document.getElementById("errorMsg");

  const file = fileInput.files[0];
  const target = targetColumnInput.value.trim();

  if (!file || !target) {
    errorMsg.textContent = "Please upload a file and specify the target column.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("target_column", target);

  showLoadingPopup();

  fetch("http://localhost:5000/run_svm", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.error) {
        errorMsg.textContent = data.error;
        return;
      }
      hideLoadingPopup();
      renderModelResults(data, "svm");
      closePopup("modelPopup");
      document.getElementById("reportPopup").style.display = "block";
    })
    .catch((err) => {
      hideLoadingPopup();
      errorMsg.textContent = "An error occurred while processing the file.";
      console.error(err);
    });
}
