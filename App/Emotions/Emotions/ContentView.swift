//
//  ContentView.swift
//  Emotions
//
//  Created by Felipe Andrade on 22/09/24.
//

import SwiftUI
import AVFoundation
import Vision
import Foundation
import CoreImage

struct ContentView: View {
    @State private var isCameraOn = true
    @State private var error: String? = nil
    
    var body: some View {
        VStack {
            if isCameraOn {
                CameraPreview()
                    .edgesIgnoringSafeArea(.all)
            } else if let error = error {
                Text(error)
            } else {
                Text("Camera is unavailable.")
            }
        }
    }
    
    func startCamera() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    self.isCameraOn = true
                } else {
                    self.error = "Camera access denied."
                }
            }
        }
    }
}


struct CameraPreview: UIViewRepresentable {
    
    class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        var classificationLabel: UILabel?
        var miniView: UIView?
        var miniImage: UIImageView?
        var parent: CameraPreview
        var currentlyAnalyzedPixelBuffer: CVPixelBuffer?
        
        lazy var expressionRequest:VNCoreMLRequest = {
            do {
                let model = try VNCoreMLModel(for: Expression(configuration: MLModelConfiguration()).model)
                let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                    self?.processExpressionRequest(for: request, error: error)
                })
                
                return request
            } catch {
                print("Error loading model: \(error)")
                fatalError("App failed to create a `VNCoreMLModel` instance.")
            }
        }()
        
        func processExpressionRequest(for request: VNRequest, error:Error?) {
            DispatchQueue.main.async {
                if let error = error {
                    print("Error in Vision request: \(error.localizedDescription)")
                    return
                }
                guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                      let firstResult = results.first,
                      let multiArray = firstResult.featureValue.multiArrayValue else {
                    print("No results or incorrect result type")
                    return
                }
                                
                // Array of class names
                let classNames = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral", "Contempt"]

                var values: [Float] = []
                for i in 0..<multiArray.count {
                    values.append(multiArray[i].floatValue)
                }

                if let maxIndex = values.firstIndex(of: values.max()!) {
                    let predictedClass = classNames[maxIndex]
                    self.classificationLabel?.text = "\(predictedClass)"
                } else {
                    print("Error: Could not find the maximum value.")
                }
            }
        }
        
        init(parent: CameraPreview) {
            self.parent = parent
        }
        
        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
            processExpression(sampleBuffer)
        }
        
        func loadCIImageFromAssets(named imageName: String) -> CIImage? {
            guard let uiImage = UIImage(named: imageName) else {
                print("Error: Image not found")
                return nil
            }
            return CIImage(image: uiImage)
        }
        
        func convertCIImageToUIImage(_ ciImage: CIImage) -> UIImage? {
            let context = CIContext(options: nil)
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                return nil
            }
            let uiImage = UIImage(cgImage: cgImage)
            return uiImage
        }
        
        func preprocessImage(_ ciImage: CIImage, size: CGSize) -> CIImage? {
            let context1 = CIContext(options: nil)
            
            guard let cgImage = context1.createCGImage(ciImage, from: ciImage.extent) else {
                return nil
            }
            
            let width = Int(round(size.width))
            let height = Int(round(size.height))
            
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                             width,
                                             height,
                                             kCVPixelFormatType_32ARGB,
                                             nil,
                                             &pixelBuffer)
            
            guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
                return nil
            }
            
            CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)
            
            let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
            let context = CGContext(data: pixelData,
                                    width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                    space: rgbColorSpace,
                                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
            
            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            
            let image = CIImage(cvPixelBuffer: pixelBuffer)
            let rotatedImage = image.oriented(.right)
            
            let grayscaleFilter = CIFilter(name: "CIColorControls")!
            grayscaleFilter.setValue(rotatedImage, forKey: kCIInputImageKey)
            grayscaleFilter.setValue(0, forKey: kCIInputSaturationKey)
            guard let grayscaleImage = grayscaleFilter.outputImage else {
                return nil
            }

            let colorMatrix = CIFilter(name: "CIColorMatrix")!
            colorMatrix.setValue(grayscaleImage, forKey: kCIInputImageKey)

            return colorMatrix.outputImage
        }

        private func processExpression(_ sampleBuffer: CMSampleBuffer) {
            DispatchQueue.main.async {
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
                let ciImageFromCameraSource = CIImage(cvPixelBuffer: pixelBuffer)
                if let processedImage = self.preprocessImage(ciImageFromCameraSource, size: CGSize(width: 48, height: 48)) {
                    self.miniImage?.image = self.convertCIImageToUIImage(processedImage)
                    let handler = VNImageRequestHandler(ciImage: processedImage)
                    do {
                        try handler.perform([self.expressionRequest])
                    } catch {
                        print("Failed to perform classification.\n\(error.localizedDescription)")
                    }
                }
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(parent: self)
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        
        let captureSession = setupCaptureSession(with: context.coordinator)
        captureSession.startRunning()
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = UIScreen.main.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        let textViewClassification = UIView(frame: CGRect(x: 50, y: 50, width: UIScreen.main.bounds.width - 100, height: 50))
        textViewClassification.layer.cornerRadius = 5
        textViewClassification.layer.masksToBounds = true
        textViewClassification.backgroundColor = .white
        textViewClassification.alpha = 0.6
        
        let miniView = UIView(frame: CGRect(x: UIScreen.main.bounds.width - 120,
                                            y: UIScreen.main.bounds.height - 170,
                                            width: 100,
                                            height: 150))
        miniView.layer.cornerRadius = 5
        miniView.layer.masksToBounds = true
        miniView.backgroundColor = .white
        miniView.alpha = 1.0
        
        context.coordinator.miniImage = UIImageView(frame: miniView.bounds)
        context.coordinator.miniImage?.contentMode = .scaleAspectFit
        context.coordinator.miniImage?.backgroundColor = .white
        miniView.addSubview(context.coordinator.miniImage!)
        
        context.coordinator.classificationLabel = UILabel(frame: textViewClassification.bounds)
        context.coordinator.classificationLabel?.text = ""
        context.coordinator.classificationLabel?.textColor = .black
        context.coordinator.classificationLabel?.font =  UIFont.systemFont(ofSize: 14)
        context.coordinator.classificationLabel?.textAlignment = .center
        textViewClassification.addSubview(context.coordinator.classificationLabel ?? UILabel())
        
        view.addSubview(textViewClassification)
        view.addSubview(miniView)
        
        return view
    }
    
    private func toggleTorch(status: Bool) {
        guard let device = AVCaptureDevice.default(for: .video) else { return }
        
        do {
            try device.lockForConfiguration()
            try device.setTorchModeOn(level: 1)
            device.torchMode = .on
            device.unlockForConfiguration()
        } catch {
            print("Error turning on flash: \(error.localizedDescription)")
        }
    }
    
    private func setupCaptureSession(with coordinator: Coordinator) -> AVCaptureSession {
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = AVCaptureSession.Preset.vga640x480
         let cameraPosition: AVCaptureDevice.Position = .front
        
        // Set up the camera as input
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: cameraPosition),
              let input = try? AVCaptureDeviceInput(device: device) else {
            return captureSession
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        // Set up the video data output
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(coordinator, queue: DispatchQueue.global(qos: .userInteractive))

        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
        }
        
        return captureSession
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
