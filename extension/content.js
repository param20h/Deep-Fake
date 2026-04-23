chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
  if (request.action === "analyze_image") {
    const { imageUrl } = request;
    
    // Attempt to download the image as a blob
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      // Show a loading notification or toast here (simplified for this example)
      console.log("Analyzing image...", imageUrl);

      const { apiBaseUrl } = await chrome.storage.sync.get(['apiBaseUrl']);
      const baseUrl = apiBaseUrl || 'http://localhost:8000';

      // Call our FastAPI backend
      const apiResponse = await fetch(`${baseUrl}/predict/image`, {
        method: "POST",
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error("Failed to analyze image");
      }

      const result = await apiResponse.json();
      
      // We can inject a visual badge over the image or show an alert
      alert(`Deepfake Analysis Result:\nFake: ${result.is_fake ? "Yes" : "No"}\nConfidence: ${(result.confidence * 100).toFixed(2)}%`);
      
    } catch (err) {
      console.error(err);
      alert("Error analyzing image. Ensure the backend is running and image is accessible.");
    }
  }
});
