%% FUNCTION extract_features
% Takes an image as input, returns a vector of extracted and computed 
% features from the image as output, including texture, edge, sharpness, 
% spectral and colour features.

function features = extract_features(I)

    % Generate a mask to ignore canvas's black margins and convert images
    % to greyscale to facilitate feature extraction
    mask = any(I>0.01, 3);
    grimg = rgb2gray(I);
   
    %%
    % Surface damage detection (texture features): grey-level co-occurency
    % matrix, extraction of contrast, homogeneity, energy, correlation 
    % (between neighbouring pixels), entropy (degree of texture's disorder)

    % Considering all spatial directions (respectively, horizontal, right
    % diagonal, vertical, left diagonal) to compare neighbouring pixels
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(grimg, 'Offset', offsets, 'Symmetric', true);
    stats = graycoprops(glcm);

    % Compute the average Shannon entropy on pixels' grey levels 
    ShannonEntropy = entropy(grimg(mask)); 

    % Texture features averaged over all four directions
    text_feat = [mean(stats.Contrast), mean(stats.Homogeneity), ...
             mean(stats.Energy), mean(stats.Correlation), ShannonEntropy];
    
    %%
    % Edge detection: extraction of edge density 
    edges = edge(grimg, 'canny') & mask;
    edge_density = sum(edges(:)) / sum(mask(:));

    %%
    % Sharpness measurement: filtering and extraction of Laplacian variance
    
    % Create a high-pass second-order filter
    h = fspecial('laplacian');
    filtered_im = imfilter(grimg, h);
    lap_values = filtered_im(mask);
    lap_var = var(double(lap_values(:)));

    %%
    % Frequency-domain analyses with FFT (power spectral features): extraction
    % of energy ratio, spectral centroid's coordinates, spectral variance
    F = fft2(grimg);
   
    % Centre low frequency components and shift high frequency components 
    % to the matrix's edges
    F = fftshift(F);
    %figure;
    %imagesc(log(1 + abs(F))); axis off;
    %colormap(jet); colorbar; title('FFT spectrum');
    P_log = log10(1 + abs(F).^2); [m,n] = size(P_log);
    spectral_var_log = var(P_log(:));

    % Compute how much spectral power is due to high frequency components
    % (exclude low frequency region, i.e. the central one)
    [x,y] = meshgrid(1:n, 1:m);
    mat_centre_distance = sqrt((x - n/2).^2 + (y - m/2).^2);
    radius = min(m,n)/4; 
    high_freqs_region = mat_centre_distance > radius; 
    energy_ratio = sum(P_log(high_freqs_region)) / sum(P_log(:));

    % Find the distance between the centroid (i.e. the point where spectral
    % power is averagely concentrated) and the spectrum's centre (which
    % coincides with the image matrix's centre)
    avg_spectral_dist = sum(mat_centre_distance(:) .* P_log(:)) / sum(P_log(:));
    norm_centroid_dist = avg_spectral_dist / sqrt((n/2)^2 + (m/2)^2);

    fft_feat = [energy_ratio, spectral_var_log, norm_centroid_dist];
    
    %%
    % Colour features: visualising histogram of pixel intensities and 
    % computing mean, standard deviation and skewness of intensity, for 
    % each RGB colour index
    R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
    tR = R(mask); tG = G(mask); tB = B(mask);
    %figure;
    %subplot(1,3,1); imhist(tR); title('R histogram');
    %subplot(1,3,2); imhist(tG); title('G histogram');
    %subplot(1,3,3); imhist(tB); title('B histogram');
   
    clr_feat = [mean(tR(:)), mean(tG(:)), mean(tB(:)), ...
        std(double(tR(:))), std(double(tG(:))), std(double(tB(:))), ...
        skewness(double(tR(:))), skewness(double(tG(:))), skewness(double(tB(:))),];

    % Combine all extracted features into a single feature vector
    features = [text_feat, edge_density, lap_var, fft_feat, clr_feat];

end
