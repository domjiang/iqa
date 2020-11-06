valid_image=0;
image_path=[];
for iter = 0:1500
    readpath=strcat('val/images/val_',int2str(iter),'.JPEG');
    

    image1 = imread(readpath);
    if length(size(image1)) == 3
        img1 = rgb2ycbcr(image1);
        img1(:,:,2) = image1(:,:,2);

        I = im2double(img1);

        %Parameter Initialisation
        [m,n,~] = size(I);
        epsilon = 0.1; 
        count = 1; 
        scale = 8;
        %Convert m x n x 3 image into [(8x8x3) x count] patches
        i = 1;
        while (i <m - (scale - 2))
            j = 1;
            while (j< n-(scale-2)) %(j < 64)
                patch_temp = I(i:i+(scale-1),j:j+(scale-1),:);       
                patches(:,count) = reshape(patch_temp,[],1);
                count = count+1;
                j = j+scale;
            end    
            i = i+scale;
        end

        % Subtract mean patch (hence zeroing the mean of the patches)
        meanPatch = mean(patches,2);  
        patches = bsxfun(@minus, patches, meanPatch);

        % Apply ZCA whitening
        sigma = patches * patches' / (count-1);
        [u, s, ~] = svd(sigma);
        ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
        patches = ZCAWhite * patches;

        writepath=strcat('processed_image/',int2str(iter),'.mat');
        image_path = [image_path,' ',writepath];

        save(writepath,'patches');
        valid_image = valid_image+1;
    end
end
disp(valid_image)

writematrix(image_path,'path.csv')
