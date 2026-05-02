%% IMAGE CONVERTER
% Purpose: converting all images in the dataset 
% 'https://www.kaggle.com/datasets/pes1ug22am047/damaged-and-undamaged-artworks?select=AI_for_Art_Restoration_2'
% to a common format, e.g. .jpg, as the dataset contains heterogeneous 
% extensions, and some files might be corrupted or unreadable as they are.

input_path = 'C:\Users\linda\Desktop\materiali università\magistrale\Computing methods for experimental physics\project dataset\raw';

files = dir(fullfile(input_path, '**', '*.*'));

converted = 0;
failed = 0;

for i = 1:length(files)
    
    % Avoid trying to read folders
    if files(i).isdir
        continue;
    end

    input_pathfile = fullfile(files(i).folder, files(i).name);

    try
        % Reading each image regardless of its format
        img = imread(input_pathfile);

        % Keeping the same name, except for the extension
        [~, name, ~] = fileparts(files(i).name);
        output_filepath = fullfile(files(i).folder, [name '.jpg']);

        % Save each image in .jpg format
        imwrite(img, output_filepath, 'jpg');

        converted = converted + 1;

    catch err
        fprintf('Failed to process file %s\n', input_pathfile);
        fprintf('Error: %s\n\n', err.message);
        failed = failed + 1;
    end
end

fprintf('\nConversion completed with\nConverted files: %d\nFailed files: %d\n', converted, failed);
