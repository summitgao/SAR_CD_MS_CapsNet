clear;
clc;
close all;

addpath('./Utils');

% PatSize 必须为奇数
PatSize = 9;
k_n = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(' ... ... read image file ... ... ... ....\n');
 im1   = imread('./pic/YellowRiverI_1.bmp');
 im2   = imread('./pic/YellowRiverI_2.bmp');
 im_gt = imread('./pic/YellowRiverI_gt.bmp');
fprintf(' ... ... read image file finished !!! !!!\n\n');

im1 = double(im1(:,:,1));
im2 = double(im2(:,:,1));
im_gt1 = double(im_gt(:,:,1));

[ylen, xlen] = size(im1);
% 求 neighborhood-based ratio image
fprintf(' ... .. compute the neighborhood ratio ..\n');
nrmap = nr(im1, im2, k_n);
nrmap = max(nrmap(:))-nrmap;
nrmap = nr_enhance( nrmap );

% 图像周围填零，然后每个像素周围取Patch，保存
mag = (PatSize-1)/2;
imTmp = zeros(ylen+PatSize-1, xlen+PatSize-1);
imTmp((mag+1):end-mag,(mag+1):end-mag) =nrmap ; 
im_gt=imTmp;

nrmap =[imTmp,imTmp,imTmp ];

nrmap=reshape(nrmap,[ylen+PatSize-1,xlen+PatSize-1,3]);
save YellowRiverI.mat nrmap

im_gt1(im_gt1==0)=1;
im_gt1(im_gt1==255)=2;
 im_gt((mag+1):end-mag,(mag+1):end-mag)=im_gt1;
 
save YellowRiverI_gt.mat im_gt
