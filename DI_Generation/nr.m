
% A neighborhood-based ratio approach for change detection in SAR image %

function nrmap = nr(im1, im2, k)
    
    [ylen, xlen] = size(im1);
    ratio = zeros(ylen, xlen);
    nrmap = zeros(ylen, xlen);

    for j = 1:ylen
        for i = 1:xlen
            if im1(j,i)>im2(j,i)
                ratio(j,i) = (im2(j,i)+0.001)/(im1(j,i)+0.001);
            elseif im1(j,i)<im2(j,i)
                ratio(j,i) = (im1(j,i)+0.001)/(im2(j,i)+0.001);
            else
                ratio(j,i) = 1;
            end
        end
    end

    for j = 1+(k-1)/2:ylen-(k-1)/2
        for i = 1+(k-1)/2:xlen-(k-1)/2        
            u = 0; diat = 0;
            smin=0; smax=0;

            im_se1 = im1(j-(k-1)/2:j+(k-1)/2, i-(k-1)/2:i+(k-1)/2);
            im_se2 = im2(j-(k-1)/2:j+(k-1)/2, i-(k-1)/2:i+(k-1)/2);
            rat_se = ratio(j-(k-1)/2:j+(k-1)/2, i-(k-1)/2:i+(k-1)/2);

            smin = im_se1.* (im_se1 <= im_se2);
            smin = smin + im_se2.* (im_se1 >  im_se2);
            smin = sum(smin(:));

            smax = im_se1.* (im_se1 >=  im_se2);
            smax = smax + im_se2.* (im_se1 < im_se2);
            smax = sum(smax(:));

            % 求均值，方差，以及lamda
            u    = mean(rat_se(:));        
            diat = var(rat_se(:));
            lmd  = (diat+0.001)/(u+0.001);

            if lmd>1
                lmd = 1;
            end

            if smax==0
                nrmap(j,i) = lmd*ratio(j,i)+(1-lmd);
            else
                nrmap(j,i) = lmd*ratio(j,i)+(1-lmd)*smin/smax;
            end          
        end
    end

    clear j i u diat rat_se ratio smax smin lmd;
    clear im_se1 im_se2;

    % 处理一下四个边上的像素
    tmp = nrmap(1+(k-1)/2:ylen-(k-1)/2, 1+(k-1)/2:xlen-(k-1)/2);
    u = mean(tmp(:));
    nrmap(1:1+(k-1)/2, :) = u;   nrmap(ylen-(k-1)/2:ylen, :) = u;
    nrmap(:, 1:1+(k-1)/2) = u;   nrmap(:, xlen-(k-1)/2:xlen) = u;

    clear u tmp;
    clear k xlen ylen;
    
end






