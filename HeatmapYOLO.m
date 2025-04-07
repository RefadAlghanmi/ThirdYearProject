classdef HeatmapYOLO < handle
    properties
        SelectedDB
        LocalPath
        HeatmapPath
        UIFig
        InputLow
        InputHigh
        ImageLabel
        DatasetSelector
    end
    
    methods
        function obj = HeatmapYOLO()
            obj.LocalPath = pwd;
            dbFiles = obj.loadDBFiles();
            
            if isempty(dbFiles)
                obj.SelectedDB = '';
            else
                obj.SelectedDB = dbFiles{1};
            end
            
            obj.HeatmapPath = '';
            obj.createGUI();
        end
        
        function dbFiles = loadDBFiles(obj)
            dbFilesStruct = dir(fullfile(obj.LocalPath, '*.db'));
            dbFiles = {dbFilesStruct.name};
        end
        
        function createGUI(obj)
            obj.UIFig = uifigure('Name', 'Heatmap Generator with YOLO', 'Position', [100, 100, 800, 600]);
            
            uilabel(obj.UIFig, 'Text', 'Enter lower range:', 'Position', [50, 550, 120, 20]);
            obj.InputLow = uieditfield(obj.UIFig, 'numeric', 'Position', [180, 550, 100, 22]);
            
            uilabel(obj.UIFig, 'Text', 'Enter higher range:', 'Position', [300, 550, 120, 20]);
            obj.InputHigh = uieditfield(obj.UIFig, 'numeric', 'Position', [430, 550, 100, 22]);
            
            obj.DatasetSelector = uidropdown(obj.UIFig, 'Items', obj.loadDBFiles(), 'Position', [50, 500, 200, 22], ...
                'ValueChangedFcn', @(src, ~) obj.updateSelectedDB(src.Value));
            
            uibutton(obj.UIFig, 'Text', 'Generate Heatmap', 'Position', [50, 450, 150, 30], ...
                'ButtonPushedFcn', @(~,~) obj.generateHeatmap());
            
            uibutton(obj.UIFig, 'Text', 'Train YOLO Model', 'Position', [220, 450, 150, 30], ...
                'ButtonPushedFcn', @(~,~) obj.trainYOLO());
            
            uibutton(obj.UIFig, 'Text', 'Run YOLO Detection', 'Position', [390, 450, 150, 30], ...
                'ButtonPushedFcn', @(~,~) obj.runYOLODetection());
            
            obj.ImageLabel = uiimage(obj.UIFig, 'Position', [50, 50, 700, 380]);
        end
        
        function updateSelectedDB(obj, selectedValue)
            obj.SelectedDB = selectedValue;
        end
        
        function generateHeatmap(obj)
            if isempty(obj.SelectedDB) || ~isfile(obj.SelectedDB)
                uialert(obj.UIFig, 'No valid database selected.', 'Error');
                return;
            end
            
            lowerRange = obj.InputLow.Value;
            higherRange = obj.InputHigh.Value;
            
            conn = sqlite(fullfile(obj.LocalPath, obj.SelectedDB), 'readonly');
            data = fetch(conn, 'SELECT * FROM data');
            close(conn);
            
            if isempty(data)
                uialert(obj.UIFig, 'No data found in the database.', 'Error');
                return;
            end
            
            dataMatrix = table2array(data)';  
            disp(dataMatrix)
            
            if size(dataMatrix, 2) <= higherRange
                uialert(obj.UIFig, 'Selected range exceeds dataset size.', 'Error');
                return;
            end
            
            subsetData = dataMatrix(:, lowerRange:higherRange);
            
            realData = subsetData(16:29, :);
            bzReal = realData(1:8, :);
            bxReal = realData(9:14, :);
            
            imaginaryData = subsetData(2:15, :);
            bzImaginary = imaginaryData(1:8, :);
            bxImaginary = imaginaryData(9:14, :);
            
            bxMagnitude = sqrt(bxReal .^ 2 + bxImaginary .^ 2);
            bzMagnitude = sqrt(bzReal .^ 2 + bzImaginary .^ 2);
            
            ref_Bz = mean(bzMagnitude, 2);
            Bz_Magnitude_Normalized = (bzMagnitude - ref_Bz) ./ ref_Bz;
            
            ref_Bx = mean(bxMagnitude, 2);
            Bx_Magnitude_Normalized  = (bxMagnitude - ref_Bx) ./ ref_Bx;
            
            figure;
            subplot(2,1,1);
            imagesc(Bx_Magnitude_Normalized);
            title('Bx Signal Magnitude Normalized');
            xlabel('Time');
            ylabel('Bx Magnitude');
            colormap('jet');
            colorbar;
            grid on;
            
            subplot(2,1,2);
            imagesc(Bz_Magnitude_Normalized);
            title('Bz Signal Magnitude Normalized');
            xlabel('Time');
            ylabel('Bz Magnitude');
            colormap('jet');
            colorbar;
            grid on;
            
            filename = sprintf('heatmap[%d,%d].png', lowerRange, higherRange);
            exportgraphics(gcf, filename, 'Resolution', 300);
            obj.HeatmapPath = filename;
            obj.ImageLabel.ImageSource = filename;
        end
        
        function trainYOLO(obj)
        % ✅ Corrected dataset path format
            yamlFile = 'C:/Users/refad/Downloads/Desktop/uni files/year 3/project/Coding/all coding/dataset.yaml';
            savePath = 'runs/detect/train'; % ✅ Ensures trained model is saved correctly

            if ~isfile(yamlFile)
                uialert(obj.UIFig, 'Dataset file not found! Check the path.', 'Error');
                return;
            end
        
            disp('🔹 Starting YOLO training...');
        
            % ✅ Fix path format using double backslashes or forward slashes
            command = sprintf(['python -c "from ultralytics import YOLO; ' ...
                'model=YOLO(''yolov8n.pt''); ' ...
                'model.train(data=r''%s'', epochs=180, imgsz=640, project=r''%s'', name=''train'',device=''cpu'')"'], ...
                yamlFile, savePath);
        
            system(command);
        
            disp('✅ YOLO Model Training Completed!');
        end
        
        function runYOLODetection(obj)
            % ✅ Define paths
            trainedModelPath = 'runs/detect/train/train/weights/best.pt';
            detectionOutput = 'runs/detect/';

            % ✅ Check if trained model exists
            if ~isfile(trainedModelPath)
                uialert(obj.UIFig, 'Trained model not found! Check runs/detect/train/weights/.', 'Error');
                return;
            end

            % ✅ Check if heatmap exists
            if isempty(obj.HeatmapPath) || ~isfile(obj.HeatmapPath)
                uialert(obj.UIFig, 'No heatmap available for detection.', 'Error');
                return;
            end
        
            disp('🔹 Running YOLO detection using trained model...');
                
            % ✅ Properly formatted command for running detection
            command = sprintf(['python -c "from ultralytics import YOLO; ' ...
                'model=YOLO(r''%s''); ' ...
                'model.predict(source=r''%s'', save=True, save_txt=True, project=r''%s'', name=''predict'', conf=0.3, ' ...
                'device=''cpu'')"'], ...
                trainedModelPath, obj.HeatmapPath, detectionOutput);
        
            system(command);

            pause(5); % ✅ Give time for YOLO to generate output
       
            % ✅ Locate the latest detection folder
            predictFolders = dir(fullfile(detectionOutput, 'predict*'));
            if isempty(predictFolders)
                uialert(obj.UIFig, 'No YOLO detection folder found.', 'Error');
                return;
            end

            % ✅ Find the latest detected image
            [~, idx] = max([predictFolders.datenum]);  
            latestPredictFolder = fullfile(predictFolders(idx).folder, predictFolders(idx).name);

            % ✅ Find the latest detected image inside that folder
            detectedFiles = dir(fullfile(latestPredictFolder, '*.jpg'));
            if isempty(detectedFiles)
                uialert(obj.UIFig, 'No detection images found.', 'Error');
                return;
            end
                    
            [~, imgIdx] = max([detectedFiles.datenum]);  
            detectedImagePath = fullfile(detectedFiles(imgIdx).folder, detectedFiles(imgIdx).name);
            
             % ✅ Force MATLAB to refresh image cache
            obj.ImageLabel.ImageSource = ''; % Clear previous image
            drawnow; % Ensure UI updates
            pause(0.5); % Small delay to force refresh
            
            
            % ✅ Display detected image in GUI
            obj.ImageLabel.ImageSource = detectedImagePath;
            drawnow; % Refresh UI
            disp('✅ YOLO Detection Completed using Trained Model!');

            % Load the heatmap image
            detectedImg = imread(detectedImagePath);
            imshow(detectedImg);
            hold on;

            % ✅ Load YOLO label files
            labelFiles = dir(fullfile(latestPredictFolder, 'labels', '*.txt'));
        
            if isempty(labelFiles)
                uialert(obj.UIFig, 'No objects detected. Try lowering the confidence threshold.', 'Error');
                return;
            end                       

            % Draw bounding boxes
            for file = labelFiles'
                filePath = fullfile(file.folder, file.name);
                detections = load(filePath);
                for i = 1:size(detections, 1)
                    x_center = detections(i, 2) * size(detectedImg, 2);
                    y_center = detections(i, 3) * size(detectedImg, 1);
                    width = detections(i, 4) * size(detectedImg, 2);             
                    height = detections(i, 5) * size(detectedImg, 1);
                    rectangle('Position', [x_center - width/2, y_center - height/2, width, height], 'EdgeColor', 'r', ...
                        'LineWidth', 2);
                end
            end
            hold off;
            

            % ✅ Save the image with detections
            outputImagePath = fullfile(latestPredictFolder, 'detected_heatmap.png');
            saveas(gcf, outputImagePath);

            disp(outputImagePath)
        
            % ✅ Display detected image in GUI
            obj.ImageLabel.ImageSource = outputImagePath;
            disp('✅ YOLO Detection Completed with Bounding Boxes!');
        end
    end
end
