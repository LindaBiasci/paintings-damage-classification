%% FUNCTION extract_features
% Takes an image as input, returns a vector of extracted and computed 
% features from the image as output, including texture, edge, sharpness, 
% spectral and colour features.

function features = extract_features(I)

    % Generate a mask to ignore canvas's black margins and convert images
    % to greyscale to facilitate feature extraction
    mask = any(I>0.01, 3);
    grimg = rgb2gray(I);
    [rows, cols] = size(grimg);
   
    %%
    % Surface damage detection (texture features): grey-level co-occurency
    % matrix, extraction of contrast, homogeneity, energy, correlation 
    % (between neighbouring pixels); local binary pattern, extraction of 
    % energy and entropy; global entropy (degree of texture's disorder)

    % Considering all spatial directions (respectively, horizontal, right
    % diagonal, vertical, left diagonal) to compare neighbouring pixels in
    % GLCM (macroscopic texture features)
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(grimg, 'Offset', offsets, 'Symmetric', true);
    stats = graycoprops(glcm);

    % Comparing neighbouring pixels in 3x3 squares for LBP statistics
    % (localised texture features)
    lbp_histogram = extractLBPFeatures(grimg, 'NumNeighbors', 8, 'Radius', 1);
    lbp_energy = sum(lbp_histogram.^2);
    lbp_entropy = -sum(lbp_histogram .* log2(lbp_histogram + eps));

    % Compute the average Shannon entropy on pixels' grey levels 
    ShannonEntropy = entropy(grimg(mask)); 

    % Texture features (GLCM averaged over all four directions)
    text_feat = [mean(stats.Contrast), mean(stats.Homogeneity), ...
             mean(stats.Energy), mean(stats.Correlation), ...
             ShannonEntropy, lbp_energy, lbp_entropy];
    
    %%
    % Edge detection: extraction of edge density 
    edges = edge(grimg, 'canny') & mask;
   
    % Divide images into 4x4 grids and retain the maximum edge density 
    % among these 16 patches
    r_size = repmat(rows/4, 1, 4); c_size = repmat(cols/4, 1, 4);
    e_patches = mat2cell(edges, r_size, c_size);
    m_patches = mat2cell(mask, r_size, c_size);
    edge_densities = cellfun(@(e, m) sum(e(:))/sum(m(:)), e_patches, m_patches);
    edge_density = max(edge_densities(:));

    %%
    % Sharpness measurement: filtering and extraction of Laplacian variance
    
    % Create a high-pass second-order filter
    h = fspecial('laplacian');
    filtered_im = imfilter(grimg, h);
    f_patches = mat2cell(filtered_im, r_size, c_size);

    % Compute maximum Laplacian variance among 16 patches as well
    lap_variances = cellfun(@(f, m) var(double(f(m))), f_patches, m_patches);
    lap_var = max(lap_variances(:));

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

    % Compute maximum spectral variance among 16 patches as well
    getSpecVar = @(g) var(reshape(log10(1 + abs(fftshift(fft2(g))).^2), [], 1));
    g_patches = mat2cell(grimg, r_size, c_size);
    spec_vars = cellfun(getSpecVar, g_patches);
    spectral_var_log = max(spec_vars(:));

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
