%% IMAGE PREPROCESSING
% Purposes: opening all images from the dataset 
% 'https://www.kaggle.com/datasets/pes1ug22am047/damaged-and-undamaged-artworks?select=AI_for_Art_Restoration_2',
% resizing them so they all have the same dimensions (without losing any 
% information), converting the greyscaled ones to RGB images so that they 
% all have the same format for further analyses, normalise their intensities 
% from a range [0, 255] to a range [0, 1] for each one of the three RGB 
% channels (i.e., from a uint8 type to a double type).

%% Defining paths
% Keeping the paired and unpaired datasets separated for further analyses

path = 'C:\Users\linda\Desktop\materiali università\magistrale\Computing methods for experimental physics\project_dataset';
in_path_paired = fullfile(path, 'raw', 'paired_dataset_art');
in_path_unpaired = fullfile(path, 'raw', 'unpaired_dataset_art');
out_path_paired = fullfile(path, 'processed', 'paired');
out_path_unpaired = fullfile(path, 'processed', 'unpaired');

files_paired = dir(fullfile(in_path_paired, '**', '*.jpg'));
files_unpaired = dir(fullfile(in_path_unpaired, '**', '*.jpg'));
files = [files_paired; files_unpaired];

%% Preprocessing of datasets

skipped = 0;
for i = 1:length(files)
    
    % Detect dataset type (paired or unpaired)
    if contains(files(i).folder, 'unpaired_dataset_art')
        in_base = in_path_unpaired;
        out_base = out_path_unpaired;
    elseif contains(files(i).folder, 'paired_dataset_art')
        in_base = in_path_paired;
        out_base = out_path_paired;
    end

    % Relative path and output path for each image to be processed
    rel_path = strrep(files(i).folder, in_base, '');
    out_path = fullfile(out_base, rel_path);
    out_file = fullfile(out_path, files(i).name);

    % If no such folder exists yet, create it
    if ~exist(out_path, 'dir')
        disp(out_path)
        mkdir(out_path);
    end

    % Read each image
    in_file = fullfile(files(i).folder, files(i).name);
    if ~isfile(in_file)
        continue;
    end
    try
        img = imread(in_file);
    catch
        skipped = skipped + 1;
        continue;
    end

    % Convert all images to the RGB format and normalise intensities
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    img = im2double(img);

    % Resize each image (e.g., 224x224 could be useful for further analyses) 
    % without changing proportions (no loss of information, no image 
    % stretching): the first dimension is brought to 224 and the second one
    % is computed consequently
    if size(img, 1) >= size(img, 2)
        img = imresize(img, [224 NaN]);
    else
        img = imresize(img, [NaN 224]);
    end
    h = size(img, 1); w = size(img, 2);

    % Create an empty black image, then centre the actual image on it
    canvas = zeros(224, 224, 3);
    row_start = floor((224 - h)/2) + 1;
    col_start = floor((224 - w)/2) + 1;

    canvas(row_start:row_start+h-1, col_start:col_start+w-1, :) = img;
    img = canvas;

    % Save each image in the new folder
    imwrite(img, out_file);

end

%% FEATURE EXTRACTION AND STATISTICAL ANALYSIS
% Purposes: extracting features from images in the paired dataset, then
% performing statistical analyses on their differences in order to point 
% out what changes the most between a damaged and an undamaged picture, and
% ultimately select the most diagnostic features to be extracted from the
% images in the unpaired dataset.

%% Feature extraction from paired dataset

dam_dir = fullfile(out_path_paired, 'damaged');
undam_dir = fullfile(out_path_paired, 'undamaged');
num_images_p = length(dir(fullfile(dam_dir, '*.jpg')));

feature_names = {'GLCMcontrast', 'GLCMhomogeneity', 'GLCMenergy', ...
    'GLCMcorrelation', 'Entropy', 'EdgeDensity', 'LaplacianVariance', ...
    'EnergyRatio', 'SpectralVariance', 'DistanceFromCentroid', ...
    'MeanR', 'MeanG', 'MeanB', 'StdR', 'StdG', 'StdB', 'SkewR', 'SkewG', 'SkewB'};

% Preallocate matrixes to be filled with extracted features
extfeat_dam = zeros(num_images_p, length(feature_names));
extfeat_undam = zeros(num_images_p, length(feature_names));

for i = 1:num_images_p
    % Construct filenames and full paths for each pair of images so that 
    % 'i-before' is matched with 'i-after' as it should
    name_d = sprintf('%dbefore.jpg', i);
    name_u = sprintf('%dafter.jpg', i);
    file_d = fullfile(dam_dir, name_d);
    file_u = fullfile(undam_dir, name_u);
    
    % Actual feature extraction, check if such image pair exists
    if isfile(file_d) && isfile(file_u)
        img_d = imread(file_d);
        img_u = imread(file_u);
        
        extfeat_dam(i, :) = extract_features(img_d);
        extfeat_undam(i, :) = extract_features(img_u);
    else
        warning('No paired images found for index %d', i);
        continue;
    end
end

% Remove empty rows due to skipped images, if any
extfeat_dam(all(extfeat_dam == 0, 2), :) = [];
extfeat_undam(all(extfeat_undam == 0, 2), :) = [];

% Convert to tabular data and compute the difference between feature values
% of each damaged and undamaged pair 
diff_matrix = extfeat_undam - extfeat_dam;
T_damaged = array2table(extfeat_dam, 'VariableNames', feature_names);
T_undamaged = array2table(extfeat_undam, 'VariableNames', feature_names);
T_diffs = array2table(diff_matrix, 'VariableNames', feature_names);

% Compute the correlation matrix on difference features and visualise it
R_mat = corr(diff_matrix);
figure;
heatmap(feature_names, feature_names, R_mat, 'Colormap', jet);
title('Correlation between difference features (paired images)');

%% Statistical analysis on paired features

% Features on R, G, B channels are highly correlated as expected, hence
% their averages can only be retained; moreover, distance from centroid is 
% extremely correlated to the energy ratio, as expected, and anti-correlated
% to the spectral variance as well, so it can be neglected
Mean_RGB = mean([T_diffs.MeanR, T_diffs.MeanG, T_diffs.MeanB], 2);
Std_RGB  = mean([T_diffs.StdR,  T_diffs.StdG,  T_diffs.StdB], 2);
Skew_RGB = mean([T_diffs.SkewR, T_diffs.SkewG, T_diffs.SkewB], 2);
T_diffs(:, {'DistanceFromCentroid', 'MeanR', 'MeanG', 'MeanB', ...
    'StdR', 'StdG', 'StdB', 'SkewR', 'SkewG', 'SkewB'}) = [];
T_diffs.Mean_RGB = Mean_RGB;
T_diffs.Std_RGB = Std_RGB;
T_diffs.Skew_RGB = Skew_RGB;
diff_matrix_update = table2array(T_diffs);
feature_names_update = T_diffs.Properties.VariableNames;

% Lilliefors normality test: if it returns h=0, i.e. p-value > 0.05, 
% the difference feature can be assumed to be normally distributed
nonGaussian_distr = zeros(1, length(feature_names_update));
for i = 1:length(feature_names_update)
    nonGaussian_distr(i) = lillietest(diff_matrix_update(:,i));
end

% Wilcoxon (non-parametric) test: appropriate significance test for 
% variabiables which are not normally distributed, if it returns p-value >
% 0.05, the difference feature can be treated as non-relevant for
% diagnostics (i.e. the median is approximately null)
p_values = zeros(1, length(feature_names_update));
for i = 1:length(feature_names_update)
    p_values(i) = signrank(diff_matrix_update(:,i));
end

% Update table of difference features
nonGaussian_distr = logical(nonGaussian_distr);
StatResults = table(feature_names_update', nonGaussian_distr', p_values', ...
    'VariableNames', {'Feature', 'non_normally_distributed', 'relevance_by_p_value'});
StatResults.Significant_Change = StatResults.relevance_by_p_value < 0.05;
disp('Statistical analysis of difference features'); disp(StatResults);

% Visualise boxplots of energy ratio and spectral variance, as they are
% both statistically significant but strongly anti-correlated 
subplot(1, 2, 1);
boxplot([T_damaged.EnergyRatio, T_undamaged.EnergyRatio], 'Labels', {'Damaged', 'Undamaged'});
grid on; title('Energy Ratio'); ylabel('Value');

subplot(1, 2, 2);
boxplot([T_damaged.SpectralVariance, T_undamaged.SpectralVariance], 'Labels', {'Damaged', 'Undamaged'});
grid on; title('Spectral Variance'); ylabel('Value');

sgtitle('Discriminative analysis: Energy Ratio vs. Spectral Variance');

%% Feature selection and extraction from unpaired dataset
% Features to be removed basing on correlation matrix, p-value and
% boxplots: GCLM contrast, homogeneity and energy (non-significant, 
% anti-correlated to entropy), laplacian variance (non-significant, 
% correlated to contrast), energy ratio (sligthly less statistically
% significant and dynamic than spectral variance)

selected_features = {'GLCMcorrelation', 'Entropy', 'EdgeDensity', ...
    'SpectralVariance', 'Mean_RGB', 'Std_RGB', 'Skew_RGB'};

unpaired_dataset = dir(fullfile(out_path_unpaired, '**', '*.jpg'));
num_images_u = length(unpaired_dataset);
all_tables = cell(num_images_u, 1);

for i = 1:num_images_u
    % Build path to the image
    img_path = fullfile(unpaired_dataset(i).folder, unpaired_dataset(i).name);
    img = imread(img_path);
    
    % Feature extraction, conversion and selection
    all_feats = extract_features(img);
    T_temp = array2table(all_feats, 'VariableNames', feature_names);
    
    T_temp.Mean_RGB = mean([T_temp.MeanR, T_temp.MeanG, T_temp.MeanB], 2);
    T_temp.Std_RGB  = mean([T_temp.StdR, T_temp.StdG, T_temp.StdB], 2);
    T_temp.Skew_RGB = mean([T_temp.SkewR, T_temp.SkewG, T_temp.SkewB], 2);
    
    T_final = T_temp(:, selected_features);
    
    % Add labels for classification: 1 if undamaged, 0 if damaged
    T_final.Label = double(contains(unpaired_dataset(i).folder, 'undamaged'));
    all_tables{i} = T_final;
end

% Assemble the final table of diagnostic features
T_unpaired = vertcat(all_tables{:});

%% Export data
% Save the features table as a .csv file for classification

csv_path = fullfile(path, 'features_manual.csv');
writetable(T_unpaired, csv_path);
