import React, { useState } from "react";
import { Upload, X, ChevronLeft, ChevronRight } from "lucide-react";

import linkedInIcon from "./assets/linked_in.svg";
import gitIcon from "./assets/git.svg";
import figmaIcon from "./assets/figma.svg";

export default function YoursOCR() {
  const [showInstructions, setShowInstructions] = useState(false);
  const [uploadMode, setUploadMode] = useState("single"); // 'single' or 'multiple'
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [processedResults, setProcessedResults] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [showNotification, setShowNotification] = useState(false);
  const [showCopyNotification, setShowCopyNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      processFiles(files);
    }
  };

  const processFiles = (files) => {
    const fileData = files.map((file) => ({
      file,
      name: file.name,
      preview: null,
      extractedText: "",
      processed: false,
    }));

    // Generate previews
    fileData.forEach((item, index) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedFiles((prev) => {
          const updated = [...prev];
          if (updated[index]) {
            updated[index].preview = reader.result;
          }
          return updated;
        });
      };
      reader.readAsDataURL(item.file);
    });

    setUploadedFiles(fileData);
    setCurrentImageIndex(0);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      processFiles(files);
    }
  };

  const handleSubmit = async () => {
    if (uploadedFiles.length === 0) {
      setNotificationMessage("You need to upload an image or file to submit");
      setShowNotification(true);
      setTimeout(() => setShowNotification(false), 3000);
      return;
    }

    setLoading(true);
    const results = [];

    try {
      // process all images
      for (let i = 0; i < uploadedFiles.length; i++) {
        const formData = new FormData();
        formData.append("file", uploadedFiles[i].file);

        try {
          const response = await fetch(
            "http://localhost:8000/api/v1/extract-text",
            {
              method: "POST",
              body: formData,
            }
          );

          const data = await response.json();

          if (data.success) {
            results.push({
              ...uploadedFiles[i],
              extractedText: data.results.full_text,
              processed: true,
            });
          } else {
            results.push({
              ...uploadedFiles[i],
              extractedText: "Failed to extract text",
              processed: true,
            });
          }
        } catch (error) {
          console.error(`Error processing ${uploadedFiles[i].name}:`, error);
          results.push({
            ...uploadedFiles[i],
            extractedText: "Error processing image",
            processed: true,
          });
        }
      }

      setProcessedResults(results);
      setNotificationMessage("All images processed successfully!");
      setShowNotification(true);
      setTimeout(() => setShowNotification(false), 3000);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to process images. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setUploadedFiles([]);
    setProcessedResults([]);
    setCurrentImageIndex(0);
  };

  const handleTextChange = (index, newText) => {
    setProcessedResults((prev) => {
      const updated = [...prev];
      updated[index].extractedText = newText;
      return updated;
    });
  };

  const handleCopyText = () => {
    if (processedResults.length > 0) {
      navigator.clipboard.writeText(
        processedResults[currentImageIndex].extractedText
      );
      setShowCopyNotification(true);
      setTimeout(() => setShowCopyNotification(false), 2000);
    }
  };

  const handleDownload = () => {
    if (processedResults.length === 0) return;

    if (processedResults.length === 1) {
      // for single file download
      const fileName = processedResults[0].name.replace(/\.[^/.]+$/, "");
      const blob = new Blob([processedResults[0].extractedText], {
        type: "text/plain",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${fileName}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } else {
      // for multiple files, create combined text
      let combinedText = "";

      processedResults.forEach((result, index) => {
        combinedText += `${"=".repeat(50)}\n`;
        combinedText += `Image ${index + 1}: ${result.name}\n`;
        combinedText += `${"=".repeat(50)}\n\n`;
        combinedText += result.extractedText;
        combinedText += "\n\n\n";
      });

      // download combined file
      const blob = new Blob([combinedText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "combined_ocr_results.txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      // download individual files
      processedResults.forEach((result) => {
        const fileName = result.name.replace(/\.[^/.]+$/, "");
        const blob = new Blob([result.extractedText], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${fileName}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });

      setNotificationMessage("All files downloaded!");
      setShowNotification(true);
      setTimeout(() => setShowNotification(false), 2000);
    }
  };

  const goToPrevious = () => {
    setCurrentImageIndex((prev) => Math.max(0, prev - 1));
  };

  const goToNext = () => {
    setCurrentImageIndex((prev) =>
      Math.min(processedResults.length - 1, prev + 1)
    );
  };

  const currentResult = processedResults[currentImageIndex];

  return (
    <div
      className="min-h-screen bg-white"
      style={{ fontFamily: "'Saira Condensed', sans-serif" }}
    >
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
        <a
          href="#about"
          className="text-xl font-semibold"
          style={{ color: "#626262" }}
        >
          ABOUT US
        </a>
        <div className="flex items-center gap-2">
          <img
            src="/images/tdt_logo.png"
            alt="TDT Logo"
            className="w-10 h-10 object-contain"
          />
          <span className="text-2xl font-bold" style={{ color: "#626262" }}>
            Yours OCR
          </span>
        </div>
        <button
          onClick={() => setShowInstructions(true)}
          className="px-6 py-2 text-white rounded"
          style={{ backgroundColor: "#626262" }}
        >
          See Instructions Here
        </button>
      </header>

      {/* Instructions Modal */}
      {showInstructions && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-start justify-center pt-20 z-50">
          <div className="bg-gray-100 rounded-lg p-8 max-w-3xl w-full mx-4 relative">
            <button
              onClick={() => setShowInstructions(false)}
              className="absolute top-4 right-4 text-gray-600 hover:text-gray-800"
            >
              <X size={24} />
            </button>
            <h2
              className="text-2xl font-bold mb-6 text-center"
              style={{ color: "#000000" }}
            >
              Instructions
            </h2>
            <div className="space-y-4 text-black">
              <p>How to Use This OCR Tool:</p>
              <p>
                1. Choose Mode: Select "Single Image" or "Multiple Images" mode.
              </p>
              <p>
                2. Upload: Click "Browse" or drag and drop your image file(s)
                (PNG, JPG, JPEG) into the upload area.
              </p>
              <p>
                3. Submit: Click the "Submit" button to process all images at
                once and extract text.
              </p>
              <p>
                4. Review: For multiple images, use arrow buttons to navigate
                between results. You can edit the extracted text directly.
              </p>
              <p>
                5. Copy or Download: Use "Copy Text" to copy current text to
                clipboard, or "Download" to save files.
              </p>
              <p>
                6. Download Options: Single image creates one .txt file.
                Multiple images create individual .txt files plus a combined
                file with all results.
              </p>
              <p>
                7. Reset: Click "Cancel" to clear and start over with new files.
              </p>
              <p>
                Privacy & Data Handling: Your uploaded images are processed on
                our secure server using custom-trained OCR models. Files are
                temporarily stored during processing and are automatically
                deleted after extraction is complete. We do not retain or share
                your uploaded content.
              </p>
              <p>
                Supported Formats: PNG, JPG, and JPEG image files. For best
                results, ensure your images are clear and well-lit with commonly
                used document fonts like Times New Roman.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Notifications */}
      {showNotification && (
        <div
          className={`fixed top-4 right-4 ${
            notificationMessage.includes("success") ||
            notificationMessage.includes("downloaded")
              ? "bg-green-500"
              : "bg-red-500"
          } text-white px-6 py-3 rounded shadow-lg z-50`}
        >
          {notificationMessage}
        </div>
      )}

      {showCopyNotification && (
        <div className="fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded shadow-lg z-50">
          ✓ Text copied to clipboard!
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-8 py-12">
        {/* Mode Selection */}
        <div className="mb-8">
          <h3 className="text-lg mb-4" style={{ color: "#000000" }}>
            Select Upload Mode
          </h3>
          <div className="flex gap-4">
            <button
              onClick={() => {
                setUploadMode("single");
                handleCancel();
              }}
              className={`px-8 py-2 rounded ${
                uploadMode === "single"
                  ? "text-white"
                  : "border border-gray-400 text-gray-700 bg-white"
              }`}
              style={
                uploadMode === "single" ? { backgroundColor: "#626262" } : {}
              }
            >
              Single Image
            </button>
            <button
              onClick={() => {
                setUploadMode("multiple");
                handleCancel();
              }}
              className={`px-8 py-2 rounded ${
                uploadMode === "multiple"
                  ? "text-white"
                  : "border border-gray-400 text-gray-700 bg-white"
              }`}
              style={
                uploadMode === "multiple" ? { backgroundColor: "#626262" } : {}
              }
            >
              Multiple Images
            </button>
          </div>
        </div>

        {/* Upload Section */}
        <div className="mb-8">
          <h3 className="text-lg mb-4" style={{ color: "#000000" }}>
            Upload your {uploadMode === "single" ? "file" : "files"} here
          </h3>
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center bg-white"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-4">
              <Upload size={48} className="text-blue-400" />
              <p className="text-gray-600">
                select your {uploadMode === "single" ? "file" : "files"} or drag
                and drop
              </p>
              <p className="text-gray-400 text-sm">PNG, JPG, JPEG accepted</p>
              <label
                className="px-8 py-2 text-white rounded cursor-pointer"
                style={{ backgroundColor: "#626262" }}
              >
                Browse
                <input
                  type="file"
                  className="hidden"
                  accept=".png,.jpg,.jpeg"
                  multiple={uploadMode === "multiple"}
                  onChange={handleFileUpload}
                />
              </label>
              {uploadedFiles.length > 0 && (
                <div className="text-sm text-gray-700 mt-2">
                  {uploadMode === "single" ? (
                    <p>Selected: {uploadedFiles[0].name}</p>
                  ) : (
                    <p>Selected: {uploadedFiles.length} file(s)</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Image Preview - Before Processing */}
        {uploadedFiles.length > 0 && processedResults.length === 0 && (
          <div className="mb-8">
            <h3 className="text-lg mb-4" style={{ color: "#000000" }}>
              Uploaded Image{uploadMode === "multiple" ? "s" : ""}
            </h3>
            <div className="border-2 border-gray-300 rounded-lg p-4 bg-white">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="flex flex-col items-center">
                    {file.preview && (
                      <img
                        src={file.preview}
                        alt={`Preview ${index + 1}`}
                        className="w-full h-32 object-cover rounded"
                      />
                    )}
                    <p className="text-xs text-gray-600 mt-2 truncate w-full text-center">
                      {file.name}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end gap-4 mb-8">
          <button
            onClick={handleCancel}
            className="px-8 py-2 border border-gray-400 rounded text-gray-700 bg-white hover:bg-gray-50"
          >
            Cancel
          </button>

          <button
            onClick={handleSubmit}
            disabled={loading || uploadedFiles.length === 0}
            className="px-8 py-2 text-white rounded disabled:opacity-50"
            style={{ backgroundColor: "#626262" }}
          >
            {loading ? "Processing..." : "Submit"}
          </button>
        </div>

        {/* Result Section - After Processing */}
        {processedResults.length > 0 && currentResult && (
          <div className="mb-12">
            {/* Navigation for multiple images */}
            {processedResults.length > 1 && (
              <div className="flex items-center justify-between mb-4">
                <button
                  onClick={goToPrevious}
                  disabled={currentImageIndex === 0}
                  className="p-2 border border-gray-400 rounded disabled:opacity-30"
                >
                  <ChevronLeft size={24} />
                </button>
                <span className="text-lg" style={{ color: "#000000" }}>
                  Image {currentImageIndex + 1} of {processedResults.length}
                </span>
                <button
                  onClick={goToNext}
                  disabled={currentImageIndex === processedResults.length - 1}
                  className="p-2 border border-gray-400 rounded disabled:opacity-30"
                >
                  <ChevronRight size={24} />
                </button>
              </div>
            )}

            {/* Current Image */}
            <div className="mb-4">
              <h3 className="text-lg mb-2" style={{ color: "#000000" }}>
                {currentResult.name}
              </h3>
              <div className="border-2 border-gray-300 rounded-lg p-4 bg-white">
                <img
                  src={currentResult.preview}
                  alt="Current preview"
                  className="max-w-full h-auto mx-auto"
                  style={{ maxHeight: "400px" }}
                />
              </div>
            </div>

            {/* Editable Text Area */}
            <div
              className="rounded-lg p-4 mb-4"
              style={{ backgroundColor: "#F4F4F4" }}
            >
              <textarea
                value={currentResult.extractedText}
                onChange={(e) =>
                  handleTextChange(currentImageIndex, e.target.value)
                }
                className="w-full min-h-[200px] bg-transparent border-none outline-none resize-y"
                style={{ color: "#000000" }}
              />
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end gap-4">
              <button
                onClick={handleCopyText}
                className="px-8 py-2 border border-gray-400 rounded text-gray-700 bg-white hover:bg-gray-50"
              >
                Copy Text
              </button>
              <button
                onClick={handleDownload}
                className="px-8 py-2 text-white rounded"
                style={{ backgroundColor: "#626262" }}
              >
                Download {processedResults.length > 1 ? "All" : "txt"}
              </button>
            </div>
          </div>
        )}

        {/* About Section */}
        <div className="mb-12" id="about">
          <h2
            className="text-2xl font-bold text-center mb-6"
            style={{ color: "#000000" }}
          >
            ABOUT THIS WEB
          </h2>
          <div className="space-y-4 text-center" style={{ color: "#000000" }}>
            <p>
              This web app is developed by a student of TDTU Information
              Technology Faculty. It aims to provide a simple and convenient way
              to extract text content from uploaded images, or to generate
              downloadable text files based on the uploaded content. This tool
              is especially useful for students, developers, and anyone who
              needs to quickly access or convert handwritten images to text
              without any complicated setup. Note: The model trained is
              especially for handwritting used in IAM handwritting dataset.
              There is no layout detection or text format detection included.
            </p>
            <p>
              There is no relation between TDTU official websites and this web
              app.
            </p>
            <p>
              This web app is developed with React and FastAPI. No data will be
              stored or shared - your uploaded images will store for a while
              during processing and are automatically deleted after extraction
              is complete.
            </p>
            <p>You can see the GitHub repository of this web app below.</p>
          </div>
        </div>

        {/* Social Icons */}
        <div className="flex justify-center gap-8 mb-12">
          <a
            href="https://www.linkedin.com/in/myat-thiri-maung-137216230/"
            className="text-gray-700 hover:text-gray-900"
          >
            <img src={linkedInIcon} alt="LinkedIn" className="w-8 h-8" />
          </a>
          <a
            href="https://github.com/MyatThiriMaung3/yours_ocr_demo"
            className="text-gray-700 hover:text-gray-900"
          >
            <img src={gitIcon} alt="GitHub" className="w-8 h-8" />
          </a>
          <a
            href="https://www.figma.com/design/E4wNXs6YEYSHvZ6LV5ULV1/TDT-OCR-WEB"
            className="text-gray-700 hover:text-gray-900"
          >
            <img src={figmaIcon} alt="Figma" className="w-8 h-8" />
          </a>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 py-6">
        <div className="max-w-7xl mx-auto px-8 flex items-center justify-between">
          <div>
            <a
              href="https://opensource.org/license/MIT"
              className="text-sm"
              style={{ color: "#000000" }}
            >
              MIT License
            </a>
            <p className="text-sm" style={{ color: "#000000" }}>
              Copyright (c) 2025
            </p>
          </div>
          <div className="flex items-center gap-2">
            <img
              src="/images/tdt_logo.png"
              alt="TDT Logo"
              className="w-10 h-10 object-contain"
            />
            <span className="text-xl font-bold" style={{ color: "#626262" }}>
              Yours OCR
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
