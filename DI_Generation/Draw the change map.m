clear all;
clc;
close all;
% load final result and the training sample index
load('training_index.mat');
load('final_result.mat');
im_gt = imread('./pic/YellowRiveI_gt.bmp');
im_gt = double(im_gt(:,:,1));

im_gt(im_gt==255)=1;
im_gt1 = im_gt';
[ylen,xlen] = size(im_gt);
im_gt_1 = reshape(im_gt1,[1,ylen*xlen]);

result = zeros(1,ylen*xlen);
index = index+1;
% the number of training samples and validation samples
train_nsamples = 1000;
validation_nsamples = 1000;
num_index = train_nsamples+validation_nsamples;

index_1 = index(:,num_index+1:end);
index_2 = index(:,1:num_index);

for i=1:length(index_1)
    result(index_1(i))=final_resule(i);
end

for j=1:length(index_2)
    result(index_2(j))=im_gt_1(index_2(j));
end

result=reshape(result,[xlen,ylen]);
result = result';

% remove small noise resigns
resign_size = 10;
 
[lab_pre,num] = bwlabel(result);
 for i = 1:num
     idx = find(lab_pre==i);
     if numel(idx) <= resign_size
        lab_pre(idx)=0;
     end
end
lab_pre = lab_pre>0;
result = uint8(lab_pre)*255;

% calculate the final result
 aa = find(im_gt==0&result~=0);
 bb = find(im_gt~=0&result==0);
 
 FA = numel(aa);
 MA = numel(bb);
 OE = FA + MA; 

CA = 1-OE/(ylen*xlen);  
imshow(result,[]);