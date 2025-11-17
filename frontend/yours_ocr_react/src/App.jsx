import React, { useState } from 'react';
import { Upload, X } from 'lucide-react';

import linkedInIcon from './assets/linked_in.svg';
import gitIcon from './assets/git.svg';
import figmaIcon from './assets/figma.svg';

export default function YoursOCR() {
  const [showInstructions, setShowInstructions] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [showNotification, setShowNotification] = useState(false);
  const [showCopyNotification, setShowCopyNotification] = useState(false);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      // Simulate text extraction
      const loremText = `Lorem ipsum dolor sit amet, frandor niven pellora clast. Dorem phelis quentur mavic troen falim venor. Plasmet erion dulav crentis haldor sit vennet. Traven pelic moran flict unden graviton selum. Ferand ulcor vinit aspor lenct dravis polen. Mvera clost inum pravel torent quevis falda. Dramet in volar crimis morven adest lora ven. Pliver toran quist falum dretor namic sela. Ventor claram ispen dorel mavin trat ulvim fren.

Lorem ipsum dolor sit amet, frandor niven pellora clast. Dorem phelis quentur mavic troen falim venor. Plasmet erion dulav crentis haldor sit vennet. Traven pelic moran flict unden graviton selum. Ferand ulcor vinit aspor lenct dravis polen. Mvera clost inum pravel torent quevis falda. Dramet in volar crimis morven adest lora ven. Pliver toran quist falum dretor namic sela. Ventor claram ispen dorel mavin trat ulvim fren.`;
      setExtractedText(loremText);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setUploadedFile(file);
      const loremText = `Lorem ipsum dolor sit amet, frandor niven pellora clast. Dorem phelis quentur mavic troen falim venor. Plasmet erion dulav crentis haldor sit vennet. Traven pelic moran flict unden graviton selum. Ferand ulcor vinit aspor lenct dravis polen. Mvera clost inum pravel torent quevis falda. Dramet in volar crimis morven adest lora ven. Pliver toran quist falum dretor namic sela. Ventor claram ispen dorel mavin trat ulvim fren.

Lorem ipsum dolor sit amet, frandor niven pellora clast. Dorem phelis quentur mavic troen falim venor. Plasmet erion dulav crentis haldor sit vennet. Traven pelic moran flict unden graviton selum. Ferand ulcor vinit aspor lenct dravis polen. Mvera clost inum pravel torent quevis falda. Dramet in volar crimis morven adest lora ven. Pliver toran quist falum dretor namic sela. Ventor claram ispen dorel mavin trat ulvim fren.`;
      setExtractedText(loremText);
    }
  };

  const handleSubmit = () => {
    if (!uploadedFile) {
      setShowNotification(true);
      setTimeout(() => setShowNotification(false), 3000);
    }
  };

  const handleCancel = () => {
    setUploadedFile(null);
    setExtractedText('');
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(extractedText);
    setShowCopyNotification(true);
    setTimeout(() => setShowCopyNotification(false), 2000);
  };

  const handleDownloadPdf = () => {
    const fileName = uploadedFile ? uploadedFile.name.replace(/\.[^/.]+$/, '') : 'document';
    const blob = new Blob([extractedText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-white" style={{ fontFamily: "'Saira Condensed', sans-serif" }}>
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
        <a href="#about" className="text-xl font-semibold" style={{ color: '#626262' }}>
          ABOUT US
        </a>
        <div className="flex items-center gap-2">
          <img 
            src="/images/tdt_logo.png" 
            alt="TDT Logo" 
            className="w-10 h-10 object-contain"
          />
          <span className="text-2xl font-bold" style={{ color: '#626262' }}>Yours OCR</span>
        </div>
        <button
          onClick={() => setShowInstructions(true)}
          className="px-6 py-2 text-white rounded"
          style={{ backgroundColor: '#626262' }}
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
            <h2 className="text-2xl font-bold mb-6 text-center" style={{ color: '#000000' }}>
              Instructions
            </h2>
            <div className="space-y-4 text-black">
              <p>How to Use This OCR Tool:</p>
              <p>1. Upload: Click "Browse" or drag and drop your image file (PNG, JPG, JPEG) into the upload area.</p>
              <p>2. Submit: Click the "Submit" button to process your image and extract text.</p>
              <p>3. Review: The extracted text will appear in the text box below.</p>
              <p>4. Copy or Download: Use "Copy Text" to copy to clipboard, or "Download txt" to save as a .txt file.</p>
              <p>5. Reset: Click "Cancel" to clear and start over with a new file.</p>
              <p>Privacy & Data Handling: Your uploaded images are processed on our secure server using custom-trained OCR models. Files are temporarily stored during processing and are automatically deleted after extraction is complete. We do not retain or share your uploaded content.</p>
              <p>Supported Formats: PNG, JPG, and JPEG image files. For best results, ensure your images are clear and well-lit with formatted commonly used fonts for documents, like Times New Roman.</p>
            </div>
          </div>
        </div>
      )}

      {/* Notification */}
      {showNotification && (
        <div className="fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded shadow-lg z-50">
          You need to upload an image or file to submit
        </div>
      )}

      {showCopyNotification && (
        <div className="fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded shadow-lg z-50">
          ✓ Text copied to clipboard!
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-8 py-12">
        {/* Upload Section */}
        <div className="mb-8">
          <h3 className="text-lg mb-4" style={{ color: '#000000' }}>Upload your file here</h3>
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center bg-white"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-4">
              <Upload size={48} className="text-blue-400" />
              <p className="text-gray-600">select your file or drag and drop</p>
              <p className="text-gray-400 text-sm">PNG, JPG, JPEG accepted</p>
              <label className="px-8 py-2 text-white rounded cursor-pointer" style={{ backgroundColor: '#626262' }}>
                Browse
                <input
                  type="file"
                  className="hidden"
                  accept=".png,.jpg,.jpeg"
                  onChange={handleFileUpload}
                />
              </label>
              {uploadedFile && (
                <p className="text-sm text-gray-700 mt-2">Selected: {uploadedFile.name}</p>
              )}
            </div>
          </div>
        </div>

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
            className="px-8 py-2 text-white rounded"
            style={{ backgroundColor: '#626262' }}
          >
            Submit
          </button>
        </div>

        {/* Result Section */}
        {extractedText && (
          <div className="mb-12">
            <div className="rounded-lg p-8 mb-4" style={{ backgroundColor: '#F4F4F4' }}>
              <p className="whitespace-pre-line" style={{ color: '#000000' }}>
                {extractedText}
              </p>
            </div>
            <div className="flex justify-end gap-4">
              <button
                onClick={handleCopyText}
                className="px-8 py-2 border border-gray-400 rounded text-gray-700 bg-white hover:bg-gray-50"
              >
                Copy Text
              </button>
              <button
                onClick={handleDownloadPdf}
                className="px-8 py-2 text-white rounded"
                style={{ backgroundColor: '#626262' }}
              >
                Download txt
              </button>
            </div>
          </div>
        )}

        {/* About Section */}
        <div className="mb-12" id="about">
          <h2 className="text-2xl font-bold text-center mb-6" style={{ color: '#000000' }}>
            ABOUT THIS WEB
          </h2>
          <div className="space-y-4 text-center" style={{ color: '#000000' }}>
            <p>This web app is developed by a student of TDTU Information Technology Faculty. It aims to provide a simple and convenient way to extract text content from uploaded images, or to generate downloadable text files based on the uploaded content. This tool is especially useful for students, developers, and anyone who needs to quickly access or convert file text without any complicated setup. Note** The model trained is only for the formatted documents with commonly used document fonts. There is no layout detection or text format detection included.</p>
            <p>There is no relation between TDTU official websites and this web app.</p>
            <p>This web app is developed with React and FastAPI . No data will be stored or shared — your uploaded images will store for a while during processing and are automatically deleted after extraction is complete.</p>
            <p>You can see the GitHub repository of this web app below.</p>
          </div>
        </div>

        {/* Social Icons */}
        <div className="flex justify-center gap-8 mb-12">
          <a href="https://www.linkedin.com/in/myat-thiri-maung-137216230/" className="text-gray-700 hover:text-gray-900">
            <img src={linkedInIcon} alt="LinkedIn" className="w-8 h-8" />
          </a>
          <a href="https://github.com/MyatThiriMaung3/yours_ocr_demo" className="text-gray-700 hover:text-gray-900">
            <img src={gitIcon} alt="GitHub" className="w-8 h-8" />
          </a>
          <a href="https://www.figma.com/design/E4wNXs6YEYSHvZ6LV5ULV1/TDT-OCR-WEB" className="text-gray-700 hover:text-gray-900">
            <img src={figmaIcon} alt="Figma" className="w-8 h-8" />
          </a>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 py-6">
        <div className="max-w-7xl mx-auto px-8 flex items-center justify-between">
          <div>
            <a href="https://opensource.org/license/MIT" className="text-sm" style={{ color: '#000000' }}>MIT License</a>
            <p className="text-sm" style={{ color: '#000000' }}>Copyright (c) 2025</p>
          </div>
          <div className="flex items-center gap-2">
            <img 
              src="/images/tdt_logo.png" 
              alt="TDT Logo" 
              className="w-10 h-10 object-contain"
            />
            <span className="text-xl font-bold" style={{ color: '#626262' }}>Yours OCR</span>
          </div>
        </div>
      </footer>
    </div>
  );
}