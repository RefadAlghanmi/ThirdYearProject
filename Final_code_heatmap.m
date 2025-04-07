%%% this is the final code for the display of ECT data in heat map  %%%%
%%%                       edited: 15/11/2024                       %%%%

%%%%%%%%%%%%%%%%%%%%%%%%% LOAD DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the data base
dataStr = 'Refad test data.db'; % Adjust the file path as needed
conn = sqlite(dataStr); % Open a connection to the database

%Fetch all data from the database
sqlquery = 'SELECT * FROM data';
data = fetch(conn, sqlquery); % Execute the SQL query
close(conn); % Close the connection4
  
% Convert the table data into an array 
data_Matrix = table2array(data);

%Transporse the matrix for further processing 
data_Matrix = data_Matrix';

%%%%%%%%%%%%%% INPUT THE RANGE WANTED FOR THE DATA %%%%%%%%%%%%%

lower_Range = input('Enter the lower range of the data [1, 35400]: ');% Prompt the user for the first input

higher_Range = input('Enter the higher range of the data [1, 35400]: '); % Prompt the user for the second input

subset_Data = data_Matrix( :,lower_Range:higher_Range); % split the data 50

%%%%%%%%%%%%%%% REAL DATA %%%%%%%%%%%%%%%%%

real_Data = subset_Data(16:29, :); % Real part of the data 
Bz_Real = real_Data(1:8,:);
Bx_Real = real_Data(9:14,:);

%%%%%%%%%%%%% IMAGINARY DATA %%%%%%%%%%%%%%

imaginary_Data = subset_Data(2:15, :); %imaginary part of the data
Bz_imaginary = imaginary_Data(1:8,:);
Bx_imaginary = imaginary_Data(9:14, :);

%%%%%%%%%%%%%   CALCULATE THE MAGNITUDE OF Bx AND Bz %%%%%%%

Bx_Magnitude = sqrt(Bx_Real.^2 + Bx_imaginary.^2);
Bz_Magnitude = sqrt(Bz_Real.^2 + Bz_imaginary.^2);

%%%%%%%%%%%%%%%%%%%  NORMALIZE THE DATA %%%%%%%%%%%%%%%%

ref_Bz = Bz_Magnitude(:, 2);
%ref_Bz = mean(Bz_Magnitude,2) ;
Bz_Magnitude_Normalized = (Bz_Magnitude - ref_Bz) ./ ref_Bz; % Subtract ref and divide for normalization


ref_Bx = Bx_Magnitude(:, 2);
%ref_Bx = mean(Bx_Magnitude,2) ;
Bx_Magnitude_Normalized  = (Bx_Magnitude - ref_Bx) ./ ref_Bx; % Subtract ref and divide for normalization

%%%%%%%%%%%%%%%%%%% PLOT THE Bx AND Bz MAGNITUDES %%%%%%%%%%%%%%%%%%

figure;
subplot(2,1,1); % Bx Magnitude plot
imagesc(Bx_Magnitude_Normalized);
title(['Bx Signal Magnitude normalized ' ...
    '']);
xlabel('Time');
ylabel('Bx Magnitude');
colormap('jet');
colorbar;
grid on;

subplot(2,1,2); % Bz Magnitude plot
imagesc(Bz_Magnitude_Normalized);
title('Bz Signal Magnitude normalized ');
xlabel('Time');
ylabel('Bz Magnitude');
colormap('jet');
colorbar;
grid on;

%%%%%%%%%%%%%%%%%% SAVING THE IMAGE %%%%%%%%%%%%%%%

%Generate the filename dynamically with the specified ranges
filename = sprintf('heatmap[%d,%d]D219355.png', lower_Range, higher_Range);

%Save the figure as a PNG image
exportgraphics(gcf, filename, 'Resolution', 300);
